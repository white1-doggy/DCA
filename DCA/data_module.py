import os
from dataclasses import dataclass
from typing import Dict, List, Type

from torch.utils.data import DataLoader

try:
    from .datasets import BaseDataset, UKB
except ImportError:
    from datasets import BaseDataset, UKB


def load_split_names(split_file_path: str, split: str) -> List[str]:
    lines = [x.strip() for x in open(split_file_path, "r").readlines() if x.strip()]
    if not any(x.startswith(("train", "val", "test")) for x in lines):
        return lines

    idxs = {k: next((i for i, v in enumerate(lines) if k in v), -1) for k in ["train", "val", "test"]}
    if min(idxs.values()) < 0:
        raise ValueError("Split file contains sectioned format but missing train/val/test markers.")

    train_i, val_i, test_i = idxs["train"], idxs["val"], idxs["test"]
    sections = {
        "train": lines[train_i + 1 : val_i],
        "val": lines[val_i + 1 : test_i],
        "test": lines[test_i + 1 :],
    }
    return sections[split]


@dataclass
class PretrainDataConfig:
    root: str
    split_file_path: str
    split: str = "train"
    batch_size: int = 1
    num_workers: int = 0
    sequence_length: int = 300
    stride_within_seq: int = 1
    stride_between_seq: float = 1.0
    contrastive: bool = False
    with_voxel_norm: bool = False
    shuffle_time_sequence: bool = False


def build_pretrain_dataloader(
    cfg: PretrainDataConfig,
    dataset_cls: Type[BaseDataset] = UKB,
) -> DataLoader:
    """
    Build a dataloader by *calling existing dataset classes* (e.g., UKB) with
    subject ids coming from `split_file_path`.

    For pretraining, labels are placeholders and not used by the training loss.
    """
    if not os.path.exists(cfg.split_file_path):
        raise FileNotFoundError(f"split_file_path not found: {cfg.split_file_path}")
    if not os.path.isdir(cfg.root):
        raise NotADirectoryError(f"dataset root not found: {cfg.root}")

    names = load_split_names(cfg.split_file_path, cfg.split)
    subject_dict: Dict[str, tuple] = {str(name): (0, 0.0) for name in names}

    dataset = dataset_cls(
        root=cfg.root,
        subject_dict=subject_dict,
        sequence_length=cfg.sequence_length,
        stride_within_seq=cfg.stride_within_seq,
        stride_between_seq=cfg.stride_between_seq,
        contrastive=cfg.contrastive,
        with_voxel_norm=cfg.with_voxel_norm,
        shuffle_time_sequence=cfg.shuffle_time_sequence,
        train=(cfg.split == "train"),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(cfg.split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    return loader
