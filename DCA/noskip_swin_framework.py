from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from .swin_unetr import SwinTransformer
except ImportError:
    from swin_unetr import SwinTransformer


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class NoSkipSwinUNETR(nn.Module):
    """
    Swin-UNETR-style encoder + decoder without skip connections.

    Input:  [B, T, 96, 96, 96] (T acts as input channels)
    Output: feature map F [B, D, 96, 96, 96]
    """

    def __init__(
        self,
        in_channels: int,
        img_size=(96, 96, 96),
        feature_size: int = 48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        emb_size: int = 256,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            patch_norm=False,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            downsample="merging",
            use_v2=False,
        )

        # x4 is [B, 16*feature_size, 3, 3, 3] for 96^3 input and patch_size=2.
        ch0 = feature_size * 16
        self.dec1 = UpBlock3D(ch0, feature_size * 8)   # 3 -> 6
        self.dec2 = UpBlock3D(feature_size * 8, feature_size * 4)  # 6 -> 12
        self.dec3 = UpBlock3D(feature_size * 4, feature_size * 2)  # 12 -> 24
        self.dec4 = UpBlock3D(feature_size * 2, feature_size)      # 24 -> 48
        self.dec5 = UpBlock3D(feature_size, feature_size)           # 48 -> 96

        self.feature_head = nn.Conv3d(feature_size, emb_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x4 = self.encoder(x, normalize=True)[-1]
        h = self.dec1(x4)
        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        h = self.dec5(h)
        return self.feature_head(h)


class WeightHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden_dim or max(32, dim // 2)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [B, V, D] -> w: [B, V, 1]
        w = self.net(f)
        return torch.softmax(w, dim=1)


class ROIPredictor(nn.Module):
    def __init__(self, dim: int, k: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, k),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ReconHead(nn.Module):
    def __init__(self, dim: int, out_channels: int) -> None:
        super().__init__()
        self.head = nn.Conv3d(dim, out_channels, kernel_size=1)

    def forward(self, f_map: torch.Tensor) -> torch.Tensor:
        return self.head(f_map)


@dataclass
class LossWeights:
    lambda_roi: float = 1.0
    lambda_consistency: float = 0.1


class FMRIRepresentationModel(nn.Module):
    """
    End-to-end framework requested in the task.
    """

    def __init__(
        self,
        time_channels: int,
        feature_dim: int = 256,
        roi_target_dim: int = 16,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.time_channels = time_channels
        self.feature_dim = feature_dim
        self.roi_target_dim = roi_target_dim

        self.encoder_decoder = NoSkipSwinUNETR(
            in_channels=time_channels,
            emb_size=feature_dim,
            use_checkpoint=use_checkpoint,
        )
        self.weight_head = WeightHead(feature_dim)
        self.predictor = ROIPredictor(feature_dim, roi_target_dim)
        self.recon_head = ReconHead(feature_dim, time_channels)

    @staticmethod
    def flatten_feature_map(f_map: torch.Tensor) -> torch.Tensor:
        # f_map: [B, D, 96,96,96] -> [B, V, D]
        return f_map.reshape(f_map.shape[0], f_map.shape[1], -1).permute(0, 2, 1)

    @staticmethod
    def flatten_mask(roi_mask: torch.Tensor) -> torch.Tensor:
        # roi_mask: [B,96,96,96] -> [B,V]
        return roi_mask.reshape(roi_mask.shape[0], -1).float()

    @staticmethod
    def aggregate_roi(f: torch.Tensor, w: torch.Tensor, roi_mask_flat: torch.Tensor) -> torch.Tensor:
        # f: [B,V,D], w: [B,V,1], roi_mask_flat: [B,V]
        w_masked = w * roi_mask_flat.unsqueeze(-1)
        denom = w_masked.sum(dim=1, keepdim=True) + 1e-6
        w_norm = w_masked / denom
        z = (w_norm * f).sum(dim=1)
        return z

    def build_roi_target(self, x: torch.Tensor, roi_mask_flat: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,96,96,96] -> x_flat [B,T,V]
        For each sample, extract ROI voxels [T,N_roi], perform PCA along voxel dimension,
        and use top-K singular values as target vector y_roi [K].
        """
        bsz, t, _, _, _ = x.shape
        x_flat = x.reshape(bsz, t, -1)
        targets = []
        for b in range(bsz):
            valid = roi_mask_flat[b] > 0.5
            if valid.sum() < 2:
                targets.append(torch.zeros(self.roi_target_dim, device=x.device, dtype=x.dtype))
                continue

            x_roi = x_flat[b, :, valid]  # [T, N_roi]
            q = min(self.roi_target_dim, x_roi.shape[0], x_roi.shape[1])
            x_center = x_roi - x_roi.mean(dim=1, keepdim=True)
            _, s, _ = torch.pca_lowrank(x_center, q=q, center=False)
            vec = torch.zeros(self.roi_target_dim, device=x.device, dtype=x.dtype)
            vec[:q] = s[:q]
            targets.append(vec)
        return torch.stack(targets, dim=0)

    @staticmethod
    def subsample_roi_mask(roi_mask_flat: torch.Tensor, keep_ratio: float = 0.7) -> torch.Tensor:
        rnd = torch.rand_like(roi_mask_flat)
        keep = (rnd < keep_ratio).float()
        return roi_mask_flat * keep

    def forward(self, x: torch.Tensor, roi_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Input may be [B,1,96,96,96,T] or [B,T,96,96,96].
        if x.ndim != 5:
            raise ValueError(f"Expected x ndim=5 after preprocessing, got {x.shape}")

        f_map = self.encoder_decoder(x)                    # [B,D,96,96,96]
        f = self.flatten_feature_map(f_map)                # [B,V,D]
        w = self.weight_head(f)                            # [B,V,1]

        roi_mask_flat = self.flatten_mask(roi_mask)        # [B,V]
        z = self.aggregate_roi(f, w, roi_mask_flat)        # [B,D]

        y_hat = self.predictor(z)                          # [B,K]
        y_roi = self.build_roi_target(x, roi_mask_flat)    # [B,K]

        x_recon = self.recon_head(f_map)                   # [B,T,96,96,96]

        roi_mask_sub = self.subsample_roi_mask(roi_mask_flat)
        z_sub = self.aggregate_roi(f, w, roi_mask_sub)

        return {
            "feature_map": f_map,
            "f": f,
            "w": w,
            "z": z,
            "y_hat": y_hat,
            "y_roi": y_roi,
            "x_recon": x_recon,
            "z_sub": z_sub,
        }

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        x: torch.Tensor,
        loss_weights: LossWeights,
    ) -> Dict[str, torch.Tensor]:
        l_recon = F.mse_loss(outputs["x_recon"], x)
        l_roi = F.mse_loss(outputs["y_hat"], outputs["y_roi"])
        l_consistency = F.mse_loss(outputs["z"], outputs["z_sub"])
        l_total = l_recon + loss_weights.lambda_roi * l_roi + loss_weights.lambda_consistency * l_consistency
        return {
            "loss_total": l_total,
            "loss_recon": l_recon,
            "loss_roi": l_roi,
            "loss_consistency": l_consistency,
        }


def preprocess_input(x: torch.Tensor) -> torch.Tensor:
    """
    Accept [B,1,96,96,96,T] and convert to [B,T,96,96,96].
    If already [B,T,96,96,96], return as is.
    """
    if x.ndim == 6:
        # [B,1,D,H,W,T] -> [B,T,D,H,W]
        x = x.squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor [B,T,D,H,W], got {x.shape}")
    return x


def infer_roi_mask_from_x(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Infer ROI mask from input sequence when explicit roi_mask is not provided.
    x should be [B,T,D,H,W]. Voxels with non-zero temporal energy are considered ROI.
    """
    if x.ndim != 5:
        raise ValueError(f"Expected [B,T,D,H,W], got {x.shape}")
    return (x.abs().sum(dim=1) > eps).float()


def train_one_epoch(
    model: FMRIRepresentationModel,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: LossWeights,
) -> Dict[str, float]:
    model.train()
    meter = {"loss_total": 0.0, "loss_recon": 0.0, "loss_roi": 0.0, "loss_consistency": 0.0}
    n = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for batch in progress:
        if "x" in batch:
            x_raw = batch["x"]
        elif "fmri_sequence" in batch:
            x_raw = batch["fmri_sequence"]
            if isinstance(x_raw, (tuple, list)):  # contrastive dataset may return (seq, random_seq)
                x_raw = x_raw[0]
        else:
            raise KeyError("Batch must contain either 'x' or 'fmri_sequence'.")

        x = preprocess_input(x_raw.to(device))

        if "roi_mask" in batch:
            roi_mask = batch["roi_mask"].to(device)
        else:
            roi_mask = infer_roi_mask_from_x(x)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, roi_mask)
        losses = model.compute_losses(outputs, x, loss_weights)
        losses["loss_total"].backward()
        optimizer.step()

        for k in meter:
            meter[k] += float(losses[k].detach().cpu())
        n += 1
        progress.set_postfix(
            total=f"{float(losses['loss_total'].detach().cpu()):.4f}",
            recon=f"{float(losses['loss_recon'].detach().cpu()):.4f}",
        )

    if n == 0:
        return meter
    return {k: v / n for k, v in meter.items()}


@torch.no_grad()
def evaluate_one_epoch(
    model: FMRIRepresentationModel,
    dataloader,
    device: torch.device,
    loss_weights: LossWeights,
) -> Dict[str, float]:
    model.eval()
    meter = {"loss_total": 0.0, "loss_recon": 0.0, "loss_roi": 0.0, "loss_consistency": 0.0}
    n = 0

    progress = tqdm(dataloader, desc="val", leave=False)
    for batch in progress:
        if "x" in batch:
            x_raw = batch["x"]
        elif "fmri_sequence" in batch:
            x_raw = batch["fmri_sequence"]
            if isinstance(x_raw, (tuple, list)):
                x_raw = x_raw[0]
        else:
            raise KeyError("Batch must contain either 'x' or 'fmri_sequence'.")

        x = preprocess_input(x_raw.to(device))
        roi_mask = batch["roi_mask"].to(device) if "roi_mask" in batch else infer_roi_mask_from_x(x)

        outputs = model(x, roi_mask)
        losses = model.compute_losses(outputs, x, loss_weights)

        for k in meter:
            meter[k] += float(losses[k].detach().cpu())
        n += 1
        progress.set_postfix(
            total=f"{float(losses['loss_total'].detach().cpu()):.4f}",
            recon=f"{float(losses['loss_recon'].detach().cpu()):.4f}",
        )

    if n == 0:
        return meter
    return {k: v / n for k, v in meter.items()}
