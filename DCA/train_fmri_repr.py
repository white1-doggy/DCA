import argparse

import torch
from torch.utils.data import DataLoader

try:
    from .noskip_swin_framework import FMRIRepresentationModel, LossWeights, train_one_epoch
    from .datasets import DummyFMRIDataset, UKB
except ImportError:
    from noskip_swin_framework import FMRIRepresentationModel, LossWeights, train_one_epoch
    from datasets import DummyFMRIDataset, UKB


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
    parser.add_argument("--dataset", type=str, default="dummy", choices=["dummy", "ukb"])
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--sequence_length", type=int, default=300)
    parser.add_argument("--stride_within_seq", type=int, default=1)
    parser.add_argument("--stride_between_seq", type=float, default=1.0)
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--with_voxel_norm", action="store_true")
    parser.add_argument("--shuffle_time_sequence", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FMRIRepresentationModel(
        time_channels=args.time_channels,
        feature_dim=args.feature_dim,
        roi_target_dim=args.roi_target_dim,
    ).to(device)

    if args.dataset == "dummy":
        dataset = DummyFMRIDataset(n_samples=args.dummy_samples, t=args.time_channels)
    else:
        # Expected subject_dict format:
        # {subject_id: (sex, target), ...}
        # Replace this placeholder with your dataset split loader.
        subject_dict = {}
        dataset = UKB(
            root=args.root,
            subject_dict=subject_dict,
            sequence_length=args.sequence_length,
            stride_within_seq=args.stride_within_seq,
            stride_between_seq=args.stride_between_seq,
            contrastive=args.contrastive,
            with_voxel_norm=args.with_voxel_norm,
            shuffle_time_sequence=args.shuffle_time_sequence,
            train=True,
        )
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
