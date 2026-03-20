# No-Skip Swin-UNETR fMRI Representation Framework

This repository contains only the requested framework modules:

- `DCA/swin_unetr.py`: Swin-UNETR backbone implementation reused for the encoder.
- `DCA/noskip_swin_framework.py`: no-skip model, ROI aggregation, PCA target, losses, and training utilities.
- `DCA/datasets.py`: your dataset classes (`BaseDataset`, `UKB`, `DummyFMRIDataset`) and sequence loading logic.
- `DCA/data_module.py`: a lightweight caller to build dataloaders from `root + split_file_path` using existing dataset classes.
- `DCA/train_fmri_repr.py`: minimal training entrypoint.

## Core design

- Input: `X ∈ [B, 1, 96, 96, 96, T]` or `[B, T, 96, 96, 96]`.
- Time dimension `T` is treated as input channels.
- Encoder-decoder uses Swin-UNETR style encoder and decoder **without skip connections**.
- Decoder output is used as voxel feature map.
- ROI aggregation uses feature-dependent voxel weights (`w = softmax(weight_head(f), dim=1)`).
- ROI supervision target is built from ROI voxels with PCA (`torch.pca_lowrank`).

## What was added per your request

We did **not rewrite your dataset class logic**. Instead, we added a caller (`data_module.py`) that:
- reads subject ids from your `split_file_path`,
- builds `subject_dict`,
- calls your existing dataset class (default `UKB`) to create dataset/dataloader.

For pretraining, labels are placeholders and not used by training loss.

## Dataset protocol (pretraining)

Each sample should return:
- `fmri_sequence`: Tensor `[1,96,96,96,T]` (or `(seq, rand_seq)` for contrastive mode).
- `roi_mask` is optional.

`train_one_epoch` can consume either:
- legacy batch key `x`, or
- new batch key `fmri_sequence`.

If `roi_mask` is missing, ROI is inferred from non-zero temporal energy in input voxels.

## Split file

Supported formats:
1. plain subject list (one subject per line)
2. sectioned file with markers `train...`, `val...`, `test...`

## Run

### Dummy
```bash
cd DCA
python train_fmri_repr.py --dataset dummy --epochs 1 --batch_size 1 --time_channels 300
```

### Real pretraining dataset with your split file
```bash
cd DCA
python train_fmri_repr.py \
  --dataset pretrain_split \
  --root /path/to/subject_folders \
  --split_file_path /path/to/split.txt \
  --split train \
  --sequence_length 300 \
  --stride_within_seq 1 \
  --stride_between_seq 1
```
