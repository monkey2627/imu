# Five-sensor waist-surface retraining

Experiment root on the server:

`/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030`

## Local source layout and data flow

The source implementation is mirrored in `pnp_workspace/mobileposer/` for inspection and reproducibility. No AMASS files, real-use recordings, processed `.pt` datasets, training checkpoints, or caches are included in this local workspace. The shell entry points below intentionally target the server paths and its `mobileposer` conda environment, so the local copy is not a standalone local training setup.

For AMASS-based MobilePoser training, the current path is:

1. `run_preprocess.sh` invokes `pnp_workspace/mobileposer/no_head_layouts.py` on the server. Its `_synthesize()` routine creates the selected IMU layout; surface-mounted sensors use `surface_imu.py` to calculate attachment-point kinematics. The processed `.pt` outputs are written under the server workspace's `pnp_workspace/data/`.
2. `validate_surface_data.py` checks the generated five-sensor files and attachment metadata.
3. `run_pnp_training.sh` runs `build_mobileposer_cache.py`, which imports `PoseDataset` from `pnp_workspace/mobileposer/data.py`. `PoseDataset` converts processed files into windows/features and targets; `PoseDataModule` handles training/validation/test batches.
4. `train_mobileposer_module.py` calls `TrainingManager` from `pnp_workspace/mobileposer/train.py`, which trains the modules in `pnp_workspace/mobileposer/models/` and combines their weights.

The same `pnp_workspace/mobileposer/` package contains the streaming and Line14 implementations used by their corresponding training scripts. Real-use and personalization preparation is implemented in the top-level preparation/training scripts; those consume server-side recordings and fitted targets.

## Fixed input contract

- Order: `lw, rw, ls, rs, waist`.
- Physical roles: `LEFT_LOWER_ARM`, `RIGHT_LOWER_ARM`,
  `LEFT_LOWER_LEG`, `RIGHT_LOWER_LEG`, `WAIST`.
- AMASS locations are SMPL skin-surface attachments, not joint positions.
- The waist attachment is face 6840, vertices 1335/3504/1767, barycentric
  weights 0.025557/0.731105/0.243338, on the anterior midline at spine1 height.
- Surface-point trajectories produce acceleration. Bone-calibrated segment
  orientations use SMPL joints 18/19/4/5/3.
- Sampling rate is 60 Hz. Non-60-Hz AMASS motions are resampled with rotation
  SLERP and linear translation interpolation.

## AMASS pretraining

All three baselines use the same paper-aligned 18 AMASS subsets and seed 42.
`DanceDB` and `HUMAN4D` may be synthesized for inspection but are not consumed
by training.

- `mobileposer_s42_amass`: all four MobilePoser modules, 60 epochs.
- `stream_s42_amass`: both-frame streaming model, mount augmentation 8 degrees,
  20,000 iterations.
- `line14_s42_amass`: C21 B0 (`ori=none`, `geom=none`) Joints + Poser chain,
  40 epochs.

## Deadbug personalization

Input comes only from the five physical sensors in
`7_死虫/1/7_v1.raw-imu.json`. The recording is timestamp-resampled to 60 Hz.
The same package's FBX is fitted to SMPL and used as pseudo-supervision. This is
deliberate same-sequence overfitting, not held-out evaluation and not independent
motion-capture ground truth.

AMASS checkpoints remain unchanged. Personalized checkpoints are written to:

- `checkpoints/mobileposer_s42_deadbug_overfit`
- `checkpoints/stream_s42_deadbug_overfit`
- `checkpoints/line14_s42_deadbug_overfit`

## Status

Run `./status.sh`. A stage is complete only when its log directory contains a
`COMPLETE` marker. Logs are under `logs/`.
