# No-Skip Swin-UNETR fMRI Representation Framework

This repository now contains only the requested training framework:

- `DCA/swin_unetr.py`: Swin-UNETR backbone implementation reused for the encoder.
- `DCA/noskip_swin_framework.py`: No-skip Swin-UNETR framework, ROI aggregation, PCA target, and losses.
- `DCA/train_fmri_repr.py`: Minimal PyTorch training entrypoint with a dummy dataset.

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

## Run a minimal example

```bash
cd DCA
python train_fmri_repr.py --epochs 1 --batch_size 1 --time_channels 300
```


## No-Skip Swin-UNETR fMRI Representation Framework

A PyTorch training framework is provided in `DCA/noskip_swin_framework.py` and `DCA/train_fmri_repr.py`.

Key properties:
- Time dimension `T` is treated as input channels (`[B, 1, 96,96,96,T] -> [B, T, 96,96,96]`).
- Swin-UNETR style encoder-decoder without skip connections.
- Decoder output is used as voxel feature map.
- Feature-dependent voxel weighting (`weight_head(f)` + softmax over voxels).
- ROI-mask weighted aggregation and ROI-level prediction target built with PCA.
- Reconstruction + ROI loss + consistency loss joint training.

Minimal run (dummy data):

```bash
cd DCA
python train_fmri_repr.py --epochs 1 --batch_size 1 --time_channels 300
```
