import argparse
import os

import torch
from torch.utils.data import DataLoader

try:
    from .datasets import DummyFMRIDataset
    from .data_module import PretrainDataConfig, build_pretrain_dataloader
    from .noskip_swin_framework import (
        Stage1Model,
        VoxelWeightNet,
        stage1_train_one_epoch,
        stage2_train_one_epoch,
        infer_weight_map,
        load_volume_template,
        load_roi_network_map,
    )
except ImportError:
    from datasets import DummyFMRIDataset
    from data_module import PretrainDataConfig, build_pretrain_dataloader
    from noskip_swin_framework import (
        Stage1Model,
        VoxelWeightNet,
        stage1_train_one_epoch,
        stage2_train_one_epoch,
        infer_weight_map,
        load_volume_template,
        load_roi_network_map,
    )


def build_loaders(args):
    if args.dataset == "dummy":
        dataset = DummyFMRIDataset(n_samples=args.dummy_samples, t=args.time_channels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        return loader

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
    return build_pretrain_dataloader(cfg_train)


def main() -> None:
    parser = argparse.ArgumentParser("Two-stage fMRI training")
    parser.add_argument("--dataset", type=str, default="dummy", choices=["dummy", "pretrain_split"])
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--split_file_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dummy_samples", type=int, default=2)

    parser.add_argument("--time_channels", type=int, default=300)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--sequence_length", type=int, default=300)
    parser.add_argument("--stride_within_seq", type=int, default=1)
    parser.add_argument("--stride_between_seq", type=float, default=1.0)
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--with_voxel_norm", action="store_true")
    parser.add_argument("--shuffle_time_sequence", action="store_true")

    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--lr_stage1", type=float, default=1e-4)
    parser.add_argument("--lr_stage2", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--num_sampled_rois", type=int, default=10)

    parser.add_argument("--bg_mask_path", type=str, required=True, help="non-background mask template path")
    parser.add_argument("--roi_template_path", type=str, required=True, help="ROI label template path, e.g., Schaefer200")
    parser.add_argument(
        "--roi_network_map_path",
        type=str,
        required=True,
        help="ROI->network index map path (7-network assignment)",
    )

    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_stage1", type=str, default="")
    parser.add_argument("--resume_stage2", type=str, default="")

    parser.add_argument("--infer_only", action="store_true")
    parser.add_argument("--infer_save_path", type=str, default="./infer_weight_map.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = build_loaders(args)

    non_bg_mask = load_volume_template(args.bg_mask_path).float().to(device)
    roi_template = load_volume_template(args.roi_template_path).long().to(device)
    roi_to_network = load_roi_network_map(args.roi_network_map_path).to(device)

    stage1 = Stage1Model(time_channels=args.time_channels, emb_size=args.feature_dim).to(device)
    weight_net = VoxelWeightNet(emb_size=args.feature_dim).to(device)

    if args.resume_stage1:
        ckpt = torch.load(args.resume_stage1, map_location=device)
        stage1.load_state_dict(ckpt["model_state_dict"])
        print(f"[Stage1] resumed from {args.resume_stage1}")

    if args.resume_stage2:
        ckpt = torch.load(args.resume_stage2, map_location=device)
        weight_net.load_state_dict(ckpt["model_state_dict"])
        print(f"[Stage2] resumed from {args.resume_stage2}")

    if args.infer_only:
        # infer from one batch and save weight matrix
        batch = next(iter(train_loader))
        x_raw = batch["fmri_sequence"] if "fmri_sequence" in batch else batch["x"]
        if isinstance(x_raw, (tuple, list)):
            x_raw = x_raw[0]
        x = x_raw.to(device)
        w = infer_weight_map(stage1, weight_net, x, non_bg_mask)
        torch.save(w.cpu(), args.infer_save_path)
        print(f"[Infer] saved weight matrix to {args.infer_save_path}")
        return

    # ===== Stage 1: reconstruction-only =====
    optimizer1 = torch.optim.Adam(stage1.parameters(), lr=args.lr_stage1, weight_decay=args.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max(args.stage1_epochs, 1), eta_min=1e-6)

    for ep in range(args.stage1_epochs):
        stats = stage1_train_one_epoch(stage1, train_loader, optimizer1, device, non_bg_mask)
        scheduler1.step()
        print(f"[Stage1][{ep+1}/{args.stage1_epochs}] recon={stats['loss_recon']:.6f}")

        torch.save({"epoch": ep, "model_state_dict": stage1.state_dict(), "optimizer_state_dict": optimizer1.state_dict()},
                   os.path.join(args.save_dir, f"stage1_epoch_{ep+1}.pt"))

    # ===== Stage 2: freeze stage1 and train weight net =====
    optimizer2 = torch.optim.Adam(weight_net.parameters(), lr=args.lr_stage2, weight_decay=args.weight_decay)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.stage2_epochs, 1), eta_min=1e-6)

    best_loss = float("inf")
    for ep in range(args.stage2_epochs):
        stats = stage2_train_one_epoch(
            stage1=stage1,
            weight_net=weight_net,
            dataloader=train_loader,
            optimizer=optimizer2,
            device=device,
            non_bg_mask=non_bg_mask,
            roi_template=roi_template,
            roi_to_network=roi_to_network,
            num_sampled_rois=args.num_sampled_rois,
            margin=args.margin,
        )
        scheduler2.step()
        print(
            f"[Stage2][{ep+1}/{args.stage2_epochs}] "
            f"loss={stats['loss_total']:.6f} intra={stats['s_intra']:.6f} inter={stats['s_inter']:.6f}"
        )

        last_path = os.path.join(args.save_dir, f"stage2_epoch_{ep+1}.pt")
        torch.save({"epoch": ep, "model_state_dict": weight_net.state_dict(), "optimizer_state_dict": optimizer2.state_dict()}, last_path)
        if stats["loss_total"] < best_loss:
            best_loss = stats["loss_total"]
            torch.save({"epoch": ep, "model_state_dict": weight_net.state_dict(), "optimizer_state_dict": optimizer2.state_dict()},
                       os.path.join(args.save_dir, "stage2_best.pt"))


if __name__ == "__main__":
    main()
