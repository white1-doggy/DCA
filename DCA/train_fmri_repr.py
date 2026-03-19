import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .noskip_swin_framework import FMRIRepresentationModel, LossWeights, train_one_epoch
except ImportError:
    from noskip_swin_framework import FMRIRepresentationModel, LossWeights, train_one_epoch


class DummyFMRIDataset(Dataset):
    """
    Minimal runnable dataset for framework validation.
    Replace with your real dataset that returns:
      - x: [1,96,96,96,T] or [T,96,96,96]
      - roi_mask: [96,96,96]
    """

    def __init__(self, n_samples: int, t: int):
        self.n_samples = n_samples
        self.t = t

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.randn(1, 96, 96, 96, self.t)
        roi_mask = (torch.rand(96, 96, 96) > 0.7).float()
        return {"x": x, "roi_mask": roi_mask}


def main() -> None:
    parser = argparse.ArgumentParser("fMRI representation learning with no-skip Swin-UNETR")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--time_channels", type=int, default=300)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--roi_target_dim", type=int, default=16)
    parser.add_argument("--lambda_roi", type=float, default=1.0)
    parser.add_argument("--lambda_consistency", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dummy_samples", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FMRIRepresentationModel(
        time_channels=args.time_channels,
        feature_dim=args.feature_dim,
        roi_target_dim=args.roi_target_dim,
    ).to(device)

    dataset = DummyFMRIDataset(n_samples=args.dummy_samples, t=args.time_channels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_weights = LossWeights(lambda_roi=args.lambda_roi, lambda_consistency=args.lambda_consistency)

    for epoch in range(args.epochs):
        stats = train_one_epoch(model, loader, optimizer, device, loss_weights)
        print(
            f"epoch={epoch + 1} "
            f"total={stats['loss_total']:.6f} "
            f"recon={stats['loss_recon']:.6f} "
            f"roi={stats['loss_roi']:.6f} "
            f"cons={stats['loss_consistency']:.6f}"
        )


if __name__ == "__main__":
    main()
