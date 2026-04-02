# Two-Stage fMRI Training (No-Skip Swin Encoder-Decoder)

This repository now uses a **two-stage** training design.

## Stage 1: Reconstruction only
- Train only reconstruction task.
- Loss is masked MSE on non-background voxels, using a provided background mask template (`--bg_mask_path`).
- Output feature map from backbone: `[B, E, 96, 96, 96]`.

## Stage 2: ROI-weight learning with frozen Stage 1
- Freeze Stage-1 weights.
- Train a new weight network `W` to produce voxel weights `[B,1,96,96,96]`.
- Use ROI template (e.g. Schaefer200) and sampled ROI ids (`--num_sampled_rois`, default 10) per step.
- Compute ROI representations by within-ROI softmax over weights and weighted sum over voxel features.
- Compute similarity margin loss using 7-network mapping (`--roi_network_map_path`):
  - `S_intra`: bottom 10% intra-network similarities
  - `S_inter`: top 10% inter-network similarities
  - `Loss = max(0, m - mean(S_intra) + mean(S_inter))`
  - ROI-network map supports both:
    - direct indexing: `map[roi_id] = network_id`
    - sequential list: line `i` means ROI `i+1`

## Inference
- Use trained Stage-2 weight network to infer whole-brain normalized weight map.

## Run training
```bash
cd DCA
python train_fmri_repr.py \
  --dataset pretrain_split \
  --root /path/to/subject_folders \
  --split_file_path /path/to/split.txt \
  --time_channels 300 \
  --feature_dim 256 \
  --stage1_epochs 10 \
  --stage2_epochs 10 \
  --lr_stage1 1e-4 \
  --lr_stage2 1e-4 \
  --bg_mask_path /path/to/non_bg_mask.npy \
  --roi_template_path /path/to/schaefer200.npy \
  --roi_network_map_path /path/to/roi_to_7network.txt \
  --num_sampled_rois 10 \
  --margin 0.2 \
  --save_dir ./checkpoints
```

## Run inference
```bash
cd DCA
python train_fmri_repr.py \
  --dataset pretrain_split \
  --root /path/to/subject_folders \
  --split_file_path /path/to/split.txt \
  --bg_mask_path /path/to/non_bg_mask.npy \
  --roi_template_path /path/to/schaefer200.npy \
  --roi_network_map_path /path/to/roi_to_7network.txt \
  --resume_stage1 ./checkpoints/stage1_epoch_10.pt \
  --resume_stage2 ./checkpoints/stage2_best.pt \
  --infer_only \
  --infer_save_path ./infer_weight_map.pt
```
