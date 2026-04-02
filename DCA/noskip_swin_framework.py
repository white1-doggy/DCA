import os
from typing import Dict, Optional, Tuple

import numpy as np
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


class NoSkipSwinEncoderDecoder(nn.Module):
    """Stage-1 backbone: encoder-decoder without skip connections."""

    def __init__(self, in_channels: int, feature_size: int = 48, emb_size: int = 256, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
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
        ch0 = feature_size * 16
        self.dec1 = UpBlock3D(ch0, feature_size * 8)
        self.dec2 = UpBlock3D(feature_size * 8, feature_size * 4)
        self.dec3 = UpBlock3D(feature_size * 4, feature_size * 2)
        self.dec4 = UpBlock3D(feature_size * 2, feature_size)
        self.dec5 = UpBlock3D(feature_size, feature_size)
        self.feature_head = nn.Conv3d(feature_size, emb_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x4 = self.encoder(x, normalize=True)[-1]
        h = self.dec1(x4)
        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        h = self.dec5(h)
        return self.feature_head(h)  # [B,E,96,96,96]


class Stage1Model(nn.Module):
    """Reconstruction-only training."""

    def __init__(self, time_channels: int, emb_size: int = 256) -> None:
        super().__init__()
        self.encoder_decoder = NoSkipSwinEncoderDecoder(in_channels=time_channels, emb_size=emb_size)
        self.recon_head = nn.Conv3d(emb_size, time_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder_decoder(x)
        recon = self.recon_head(feat)
        return {"feat": feat, "recon": recon}


class VoxelWeightNet(nn.Module):
    def __init__(self, emb_size: int) -> None:
        super().__init__()
        hidden = max(32, emb_size // 2)
        self.net = nn.Sequential(
            nn.Conv3d(emb_size, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)  # logits [B,1,D,H,W]


def preprocess_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 6:
        x = x.squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
    if x.ndim != 5:
        raise ValueError(f"Expected [B,T,D,H,W], got {x.shape}")
    return x


def load_volume_template(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".pt"):
        return torch.load(path).long()
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path)).long()
    if path.endswith(".nii") or path.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError("Please install nibabel to load NIfTI template") from e
        return torch.from_numpy(nib.load(path).get_fdata()).long()
    raise ValueError("Unsupported template format")


def load_roi_network_map(path: str) -> torch.Tensor:
    if path.endswith(".pt"):
        arr = torch.load(path)
        return arr.long().view(-1)
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path)).long().view(-1)
    # txt/csv: one integer per line
    vals = [int(x.strip()) for x in open(path, "r").readlines() if x.strip()]
    return torch.tensor(vals, dtype=torch.long)


def resolve_roi_network_ids(sampled_roi_ids: torch.Tensor, roi_to_network: torch.Tensor) -> torch.Tensor:
    """
    Support two ROI->network map formats:
    1) direct-index: map[roi_id] = network_id (length >= max_roi_id+1, map[0] optional)
    2) sequential: map[i] = network_id for roi_id=i+1 (length == num_rois)
    """
    if sampled_roi_ids.numel() == 0:
        return sampled_roi_ids.new_empty((0,), dtype=torch.long)

    max_roi_id = int(sampled_roi_ids.max().item())
    if roi_to_network.numel() > max_roi_id:
        return roi_to_network[sampled_roi_ids.long()].long()

    if roi_to_network.numel() == max_roi_id:
        return roi_to_network[(sampled_roi_ids.long() - 1).clamp_min(0)].long()

    raise ValueError(
        f"roi_to_network length {roi_to_network.numel()} is incompatible with sampled max roi id {max_roi_id}. "
        "Use either direct-index map (len > max_roi_id) or sequential map (len == max_roi_id)."
    )


def masked_recon_loss(recon: torch.Tensor, x: torch.Tensor, non_bg_mask: torch.Tensor) -> torch.Tensor:
    # recon/x: [B,T,D,H,W], mask: [D,H,W] or [B,D,H,W], where 1 means keep
    if non_bg_mask.ndim == 3:
        non_bg_mask = non_bg_mask.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    mask = non_bg_mask.unsqueeze(1).float()  # [B,1,D,H,W]
    diff2 = ((recon - x) ** 2) * mask
    denom = mask.sum() * x.shape[1] + 1e-6
    return diff2.sum() / denom


def freeze_stage1(stage1: Stage1Model) -> None:
    for p in stage1.parameters():
        p.requires_grad = False
    stage1.eval()


def sample_roi_ids(roi_template: torch.Tensor, non_bg_mask: torch.Tensor, num_sampled_rois: int, device: torch.device) -> torch.Tensor:
    # roi_template: [D,H,W] integer labels, >0 valid ROI ids
    valid = (roi_template > 0) & (non_bg_mask > 0)
    roi_ids = torch.unique(roi_template[valid])
    roi_ids = roi_ids[roi_ids > 0]
    if roi_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    n = min(int(roi_ids.numel()), num_sampled_rois)
    perm = torch.randperm(int(roi_ids.numel()), device=roi_ids.device)[:n]
    return roi_ids.to(device)[perm]


def compute_roi_representations(
    feat: torch.Tensor,           # [B,E,D,H,W]
    weight_logits: torch.Tensor,  # [B,1,D,H,W]
    roi_template: torch.Tensor,   # [D,H,W]
    non_bg_mask: torch.Tensor,    # [D,H,W]
    sampled_roi_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns roi_rep [B,R,E], norm_weight [B,1,D,H,W]
    bsz, emb, d, h, w = feat.shape
    v = d * h * w
    feat_flat = feat.reshape(bsz, emb, v).transpose(1, 2)        # [B,V,E]
    w_flat = weight_logits.reshape(bsz, v)                        # [B,V]

    valid_flat = (non_bg_mask.reshape(v) > 0)
    exp_w = torch.exp(w_flat)
    exp_w = exp_w * valid_flat.unsqueeze(0)
    norm_w_flat = exp_w / (exp_w.sum(dim=1, keepdim=True) + 1e-6)

    roi_flat = roi_template.reshape(v)
    reps = []
    for roi_id in sampled_roi_ids:
        idx = (roi_flat == roi_id) & valid_flat
        if idx.sum() == 0:
            reps.append(torch.zeros((bsz, emb), device=feat.device, dtype=feat.dtype))
            continue
        roi_logits = w_flat[:, idx]                 # [B,N]
        alpha = torch.softmax(roi_logits, dim=1)    # within-ROI softmax
        roi_feat = feat_flat[:, idx, :]             # [B,N,E]
        rep = (alpha.unsqueeze(-1) * roi_feat).sum(dim=1)
        reps.append(rep)

    if len(reps) == 0:
        roi_rep = torch.zeros((bsz, 0, emb), device=feat.device, dtype=feat.dtype)
    else:
        roi_rep = torch.stack(reps, dim=1)          # [B,R,E]

    norm_weight = norm_w_flat.reshape(bsz, 1, d, h, w)
    return roi_rep, norm_weight


def compute_similarity_margin_loss(
    roi_rep: torch.Tensor,            # [B,R,E]
    sampled_roi_ids: torch.Tensor,    # [R]
    roi_to_network: torch.Tensor,     # [N_roi_max+1] or [N_roi]
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if roi_rep.shape[1] == 0:
        z = torch.tensor(0.0, device=roi_rep.device)
        return z, z, z

    rep = F.normalize(roi_rep, dim=-1)
    sim = torch.matmul(rep, rep.transpose(1, 2))  # [B,R,R]

    # map ROI id -> network id
    net_ids = resolve_roi_network_ids(sampled_roi_ids, roi_to_network)  # [R]
    unique_nets = torch.unique(net_ids)

    loss_list, intra_list, inter_list = [], [], []
    for net in unique_nets:
        in_idx = torch.where(net_ids == net)[0]
        out_idx = torch.where(net_ids != net)[0]
        if in_idx.numel() < 2 or out_idx.numel() < 1:
            continue

        # intra: all pairs in same network (upper triangle)
        intra_mat = sim[:, in_idx][:, :, in_idx]  # [B,ni,ni]
        triu = torch.triu_indices(in_idx.numel(), in_idx.numel(), offset=1, device=sim.device)
        intra_vals = intra_mat[:, triu[0], triu[1]]

        # inter: pairs across in/out
        inter_vals = sim[:, in_idx][:, :, out_idx].reshape(sim.shape[0], -1)

        if intra_vals.numel() == 0 or inter_vals.numel() == 0:
            continue

        k_intra = max(1, int(0.1 * intra_vals.shape[1]))
        k_inter = max(1, int(0.1 * inter_vals.shape[1]))

        s_intra = torch.topk(intra_vals, k=k_intra, dim=1, largest=False).values.mean(dim=1)  # lowest 10%
        s_inter = torch.topk(inter_vals, k=k_inter, dim=1, largest=True).values.mean(dim=1)   # highest 10%

        loss_net = F.relu(margin - s_intra + s_inter)
        loss_list.append(loss_net)
        intra_list.append(s_intra)
        inter_list.append(s_inter)

    if len(loss_list) == 0:
        z = torch.tensor(0.0, device=roi_rep.device)
        return z, z, z

    loss = torch.cat(loss_list).mean()
    s_intra_mean = torch.cat(intra_list).mean()
    s_inter_mean = torch.cat(inter_list).mean()
    return loss, s_intra_mean, s_inter_mean


def stage1_train_one_epoch(
    model: Stage1Model,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    non_bg_mask: torch.Tensor,
) -> Dict[str, float]:
    model.train()
    meter = {"loss_recon": 0.0}
    n = 0
    pbar = tqdm(dataloader, desc="stage1-train", leave=False)
    for batch in pbar:
        x_raw = batch["fmri_sequence"] if "fmri_sequence" in batch else batch["x"]
        if isinstance(x_raw, (tuple, list)):
            x_raw = x_raw[0]
        x = preprocess_input(x_raw.to(device))

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = masked_recon_loss(out["recon"], x, non_bg_mask)
        loss.backward()
        optimizer.step()

        meter["loss_recon"] += float(loss.detach().cpu())
        n += 1
        pbar.set_postfix(recon=f"{float(loss.detach().cpu()):.4f}")

    return {k: (v / max(n, 1)) for k, v in meter.items()}


@torch.no_grad()
def stage1_extract_feature(model: Stage1Model, x: torch.Tensor) -> torch.Tensor:
    return model(x)["feat"]


def stage2_train_one_epoch(
    stage1: Stage1Model,
    weight_net: VoxelWeightNet,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    non_bg_mask: torch.Tensor,
    roi_template: torch.Tensor,
    roi_to_network: torch.Tensor,
    num_sampled_rois: int = 10,
    margin: float = 0.2,
) -> Dict[str, float]:
    freeze_stage1(stage1)
    weight_net.train()

    meter = {"loss_total": 0.0, "s_intra": 0.0, "s_inter": 0.0}
    n = 0
    pbar = tqdm(dataloader, desc="stage2-train", leave=False)

    for batch in pbar:
        x_raw = batch["fmri_sequence"] if "fmri_sequence" in batch else batch["x"]
        if isinstance(x_raw, (tuple, list)):
            x_raw = x_raw[0]
        x = preprocess_input(x_raw.to(device))

        with torch.no_grad():
            feat = stage1_extract_feature(stage1, x)  # [B,E,D,H,W]

        optimizer.zero_grad(set_to_none=True)
        weight_logits = weight_net(feat)

        sampled_roi_ids = sample_roi_ids(roi_template, non_bg_mask, num_sampled_rois, device=device)
        roi_rep, _ = compute_roi_representations(feat, weight_logits, roi_template, non_bg_mask, sampled_roi_ids)

        loss, s_intra, s_inter = compute_similarity_margin_loss(
            roi_rep, sampled_roi_ids, roi_to_network=roi_to_network, margin=margin
        )
        loss.backward()
        optimizer.step()

        meter["loss_total"] += float(loss.detach().cpu())
        meter["s_intra"] += float(s_intra.detach().cpu())
        meter["s_inter"] += float(s_inter.detach().cpu())
        n += 1
        pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", intra=f"{float(s_intra.detach().cpu()):.4f}", inter=f"{float(s_inter.detach().cpu()):.4f}")

    return {k: (v / max(n, 1)) for k, v in meter.items()}


@torch.no_grad()
def infer_weight_map(
    stage1: Stage1Model,
    weight_net: VoxelWeightNet,
    x: torch.Tensor,
    non_bg_mask: torch.Tensor,
) -> torch.Tensor:
    freeze_stage1(stage1)
    weight_net.eval()
    x = preprocess_input(x)
    feat = stage1_extract_feature(stage1, x)
    logits = weight_net(feat)

    bsz, _, d, h, w = logits.shape
    v = d * h * w
    flat = logits.reshape(bsz, v)
    valid = (non_bg_mask.reshape(v) > 0).to(flat.device)
    exp_w = torch.exp(flat) * valid.unsqueeze(0)
    norm = exp_w / (exp_w.sum(dim=1, keepdim=True) + 1e-6)
    return norm.reshape(bsz, 1, d, h, w)
