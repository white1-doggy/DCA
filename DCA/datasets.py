import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset(Dataset):
    """
    Generic fMRI sequence loader supporting multiple datasets.

    Required kwargs include (at minimum):
      - root
      - subject_dict
      - sequence_length
      - stride_within_seq
      - stride_between_seq

    Optional kwargs (defaults shown below):
      - contrastive=False
      - with_voxel_norm=False
      - shuffle_time_sequence=False
      - train=True
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)
        self.data = self._set_data(self.root, self.subject_dict)

    def register_args(self, **kwargs):
        defaults = {
            "contrastive": False,
            "with_voxel_norm": False,
            "shuffle_time_sequence": False,
            "train": True,
        }
        defaults.update(kwargs)
        for name, value in defaults.items():
            setattr(self, name, value)
        self.kwargs = defaults

    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None):
        if self.contrastive:
            num_frames = len(os.listdir(subject_path)) - 2
            y = []
            load_fnames = [
                f"frame_{frame}.pt"
                for frame in range(start_frame, start_frame + sample_duration, self.stride_within_seq)
            ]
            if self.with_voxel_norm:
                load_fnames += ["voxel_mean.pt", "voxel_std.pt"]

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).permute(3, 2, 1, 0).unsqueeze(0)
                y.append(y_loaded)
            y = torch.cat(y, dim=4)

            random_y = []
            full_range = np.arange(0, num_frames - sample_duration + 1)
            # exclude overlapping sub-sequences within a subject
            exclude_range = np.arange(start_frame - sample_duration, start_frame + sample_duration)
            available_choices = np.setdiff1d(full_range, exclude_range)
            random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]

            load_fnames = [
                f"frame_{frame}.pt"
                for frame in range(
                    random_start_frame,
                    random_start_frame + sample_duration,
                    self.stride_within_seq,
                )
            ]
            if self.with_voxel_norm:
                load_fnames += ["voxel_mean.pt", "voxel_std.pt"]

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).permute(3, 2, 1, 0).unsqueeze(0)
                random_y.append(y_loaded)
            random_y = torch.cat(random_y, dim=4)
            return y, random_y

        # without contrastive learning
        y = []
        if self.shuffle_time_sequence:
            indices = random.sample(list(range(0, num_frames)), sample_duration // self.stride_within_seq)
            load_fnames = [f"frame_{frame}.pt" for frame in indices]
        else:
            load_fnames = [
                f"frame_{frame}.pt"
                for frame in range(start_frame, start_frame + sample_duration, self.stride_within_seq)
            ]

        if self.with_voxel_norm:
            load_fnames += ["voxel_mean.pt", "voxel_std.pt"]

        for fname in load_fnames:
            img_path = os.path.join(subject_path, fname)
            y_i = torch.load(img_path).permute(3, 2, 1, 0).unsqueeze(0)
            y.append(y_i)
        y = torch.cat(y, dim=4)
        return y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")


class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = root
        length = len(subject_dict)

        for i, subject in tqdm(enumerate(subject_dict), total=length, desc="prepare dict"):
            sex, target = subject_dict[subject]
            subject_path = os.path.join(img_root, f"{subject}")

            if not os.path.exists(subject_path):
                continue
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.sample_duration, target, sex)
                data.append(data_tuple)

        if self.train:
            self.target_values = np.array([tup[5] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sample_duration, target, sex = self.data[index]

        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sample_duration)

            background_value = y.flatten()[0]
            y = y.permute(0, 4, 1, 2, 3)
            y = torch.nn.functional.pad(y, (11, 12, 3, 3, 11, 12), value=float(background_value))
            y = y.permute(0, 2, 3, 4, 1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0, 4, 1, 2, 3)
            rand_y = torch.nn.functional.pad(rand_y, (11, 12, 3, 3, 11, 12), value=float(background_value))
            rand_y = rand_y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            }

        y = self.load_sequence(subject_path, start_frame, sample_duration)

        background_value = y.flatten()[0]
        y = y.permute(0, 4, 1, 2, 3)
        y = torch.nn.functional.pad(y, (11, 12, 3, 3, 11, 12), value=float(background_value))
        y = y.permute(0, 2, 3, 4, 1)

        return {
            "fmri_sequence": y,
            "subject_name": subject,
            "target": target,
            "TR": start_frame,
            "sex": sex,
        }


class DummyFMRIDataset(Dataset):
    """Fallback dataset that follows the same output protocol as UKB.__getitem__."""

    def __init__(self, n_samples: int, t: int):
        self.n_samples = n_samples
        self.t = t

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = torch.randn(1, 96, 96, 96, self.t)
        return {
            "fmri_sequence": x,
            "subject_name": f"dummy_{idx}",
            "target": 0.0,
            "TR": 0,
            "sex": 0,
        }


class PretrainSplitDataset(BaseDataset):
    """
    Dataset for pretraining with externally provided dataset path + split file path.
    Returns only data (no labels required for pretraining).
    """

    def __init__(self, root: str, split_file_path: str, split: str = "train", **kwargs):
        self.split_file_path = split_file_path
        self.split = split
        subjects = self._load_subjects(split_file_path, split)
        # label placeholders (unused in pretraining)
        subject_dict = {str(s): (0, 0.0) for s in subjects}
        super().__init__(root=root, subject_dict=subject_dict, **kwargs)

    @staticmethod
    def _load_subjects(split_file_path: str, split: str) -> List[str]:
        lines = [x.strip() for x in open(split_file_path, "r").readlines() if x.strip()]
        # Supports either:
        # 1) plain subject list
        # 2) sectioned file: train_subjects / val_subjects / test_subjects
        if not any(x.startswith(("train", "val", "test")) for x in lines):
            return lines

        split_markers = {
            "train": "train",
            "val": "val",
            "test": "test",
        }
        target_marker = split_markers.get(split, "train")
        idxs = {k: np.argmax([k in line for line in lines]) for k in ["train", "val", "test"]}
        train_i, val_i, test_i = idxs["train"], idxs["val"], idxs["test"]
        sections = {
            "train": lines[train_i + 1 : val_i],
            "val": lines[val_i + 1 : test_i],
            "test": lines[test_i + 1 :],
        }
        return sections[target_marker]

    def _set_data(self, root, subject_dict):
        data = []
        subjects = list(subject_dict.keys())
        for subject in tqdm(subjects, total=len(subjects), desc=f"build {self.split} set"):
            subject_path = os.path.join(root, f"{subject}")
            if not os.path.exists(subject_path):
                continue
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data.append((subject, subject_path, start_frame, self.sample_duration))
        return data

    def __getitem__(self, index):
        subject, subject_path, start_frame, sample_duration = self.data[index]
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sample_duration)
            bg = float(y.flatten()[0])
            y = torch.nn.functional.pad(y.permute(0, 4, 1, 2, 3), (11, 12, 3, 3, 11, 12), value=bg).permute(0, 2, 3, 4, 1)
            bg2 = float(rand_y.flatten()[0])
            rand_y = torch.nn.functional.pad(
                rand_y.permute(0, 4, 1, 2, 3), (11, 12, 3, 3, 11, 12), value=bg2
            ).permute(0, 2, 3, 4, 1)
            return {"fmri_sequence": (y, rand_y), "subject_name": subject}

        y = self.load_sequence(subject_path, start_frame, sample_duration)
        bg = float(y.flatten()[0])
        y = torch.nn.functional.pad(y.permute(0, 4, 1, 2, 3), (11, 12, 3, 3, 11, 12), value=bg).permute(0, 2, 3, 4, 1)
        return {"fmri_sequence": y}
