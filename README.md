# No-Skip Swin-UNETR fMRI Representation Framework

This repository contains only the requested framework modules:

- `DCA/swin_unetr.py`: Swin-UNETR backbone implementation reused for the encoder.
- `DCA/noskip_swin_framework.py`: no-skip model, ROI aggregation, PCA target, losses, and training utilities.
- `DCA/datasets.py`: unified dataset loading API (`BaseDataset`, `UKB`, `DummyFMRIDataset`).
- `DCA/train_fmri_repr.py`: minimal training entrypoint.

## Core design

- Input: `X ∈ [B, 1, 96, 96, 96, T]` or `[B, T, 96, 96, 96]`.
- Time dimension `T` is treated as input channels.
- Encoder-decoder uses Swin-UNETR style encoder and decoder **without skip connections**.
- Decoder output is used as voxel feature map.
- ROI aggregation uses feature-dependent voxel weights (`w = softmax(weight_head(f), dim=1)`).
- ROI supervision target is built from ROI voxels with PCA (`torch.pca_lowrank`).
- Training loss:
  - reconstruction loss,
  - ROI prediction loss,
  - consistency loss.

## Dataset protocol

All datasets should return a dictionary. Supported fields:
- `fmri_sequence`: Tensor `[1,96,96,96,T]` (or `(seq, rand_seq)` for contrastive mode).
- optional `roi_mask`: Tensor `[96,96,96]`.
- optional metadata (`subject_name`, `target`, `TR`, `sex`, ...).

`train_one_epoch` can consume either:
- legacy batch key `x`, or
- new batch key `fmri_sequence`.

If `roi_mask` is missing, ROI is inferred from non-zero temporal energy in input voxels.

## Run

### Dummy
```bash
cd DCA
python train_fmri_repr.py --dataset dummy --epochs 1 --batch_size 1 --time_channels 300
```

### UKB-style loader
```bash
cd DCA
python train_fmri_repr.py \
  --dataset ukb \
  --root /path/to/ukb_frames \
  --sequence_length 300 \
  --stride_within_seq 1 \
  --stride_between_seq 1.0
```
