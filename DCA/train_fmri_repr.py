import argparse
import os

import torch
from torch.utils.data import DataLoader

try:
    from .noskip_swin_framework import (
        FMRIRepresentationModel,
        LossWeights,
        train_one_epoch,
        evaluate_one_epoch,
        load_roi_template,
    )
    from .datasets import DummyFMRIDataset
    from .data_module import PretrainDataConfig, build_pretrain_dataloader
except ImportError:
    from noskip_swin_framework import (
        FMRIRepresentationModel,
        LossWeights,
        train_one_epoch,
        evaluate_one_epoch,
        load_roi_template,
    )
    from datasets import DummyFMRIDataset
    from data_module import PretrainDataConfig, build_pretrain_dataloader


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
    parser.add_argument("--dataset", type=str, default="dummy", choices=["dummy", "pretrain_split"])
    parser.add_argument("--root", type=str, default="", help="dataset root path, e.g. /path/to/subject_folders")
    parser.add_argument("--split_file_path", type=str, default="", help="path to split file")
    parser.add_argument("--sequence_length", type=int, default=300)
    parser.add_argument("--stride_within_seq", type=int, default=1)
    parser.add_argument("--stride_between_seq", type=float, default=1.0)
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--with_voxel_norm", action="store_true")
    parser.add_argument("--shuffle_time_sequence", action="store_true")
    parser.add_argument("--val_interval", type=int, default=1, help="run validation every N epochs")
    parser.add_argument("--roi_template_path", type=str, default="", help="path to ROI template (.pt/.npy/.nii/.nii.gz)")
    parser.add_argument("--num_sampled_rois", type=int, default=10, help="number of random ROIs sampled each step")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FMRIRepresentationModel(
        time_channels=args.time_channels,
        feature_dim=args.feature_dim,
        roi_target_dim=args.roi_target_dim,
    ).to(device)
    roi_template = load_roi_template(args.roi_template_path).to(device) if args.roi_template_path else None

    if args.dataset == "dummy":
        train_dataset = DummyFMRIDataset(n_samples=args.dummy_samples, t=args.time_channels)
        val_dataset = DummyFMRIDataset(n_samples=max(1, args.dummy_samples // 2), t=args.time_channels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        if not args.root or not args.split_file_path:
            raise ValueError("For pretrain_split, please provide --root and --split_file_path.")
        cfg_train = PretrainDataConfig(
            root=args.root,
            split_file_path=args.split_file_path,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sequence_length=args.sequence_length,
            stride_within_seq=args.stride_within_seq,
            stride_between_seq=args.stride_between_seq,
            contrastive=args.contrastive,
            with_voxel_norm=args.with_voxel_norm,
            shuffle_time_sequence=args.shuffle_time_sequence,
        )
        train_loader = build_pretrain_dataloader(cfg_train)

        cfg_val = PretrainDataConfig(
            root=args.root,
            split_file_path=args.split_file_path,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sequence_length=args.sequence_length,
            stride_within_seq=args.stride_within_seq,
            stride_between_seq=args.stride_between_seq,
            contrastive=args.contrastive,
            with_voxel_norm=args.with_voxel_norm,
            shuffle_time_sequence=args.shuffle_time_sequence,
        )
        try:
            val_loader = build_pretrain_dataloader(cfg_val)
        except Exception as e:
            print(f"[Warning] validation loader disabled: {e}")
            val_loader = None

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.eta_min,
        )

    loss_weights = LossWeights(lambda_roi=args.lambda_roi, lambda_consistency=args.lambda_consistency)
    os.makedirs(args.save_dir, exist_ok=True)

    start_epoch = 0
    best_val = float("inf")
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[Checkpoint] resumed from {args.resume_checkpoint} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            roi_template=roi_template,
            num_sampled_rois=args.num_sampled_rois,
        )
        log = (
            f"epoch={epoch + 1} "
            f"train_total={train_stats['loss_total']:.6f} "
            f"train_recon={train_stats['loss_recon']:.6f} "
            f"train_roi={train_stats['loss_roi']:.6f} "
            f"train_cons={train_stats['loss_consistency']:.6f}"
        )

        if val_loader is not None and ((epoch + 1) % args.val_interval == 0):
            val_stats = evaluate_one_epoch(
                model,
                val_loader,
                device,
                loss_weights,
                roi_template=roi_template,
                num_sampled_rois=args.num_sampled_rois,
            )
            log += (
                f" | val_total={val_stats['loss_total']:.6f} "
                f"val_recon={val_stats['loss_recon']:.6f} "
                f"val_roi={val_stats['loss_roi']:.6f} "
                f"val_cons={val_stats['loss_consistency']:.6f}"
            )
            current_val = val_stats["loss_total"]
            if current_val < best_val:
                best_val = current_val
                best_path = os.path.join(args.save_dir, "best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        "best_val": best_val,
                        "args": vars(args),
                    },
                    best_path,
                )
                log += " | best_saved=1"
        print(log)

        if scheduler is not None:
            scheduler.step()

        if ((epoch + 1) % args.save_every) == 0:
            latest_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_val": best_val,
                    "args": vars(args),
                },
                latest_path,
            )


if __name__ == "__main__":
    main()
