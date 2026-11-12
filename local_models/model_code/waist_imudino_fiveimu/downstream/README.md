# Five-IMU downstream training

Run `bash downstream/run_full.sh` from the server repository. The script first extracts frozen IMUDINO features for all 18 AMASS training subsets and BirdDog, then trains a pose head for 30 epochs. It uses the `imudino` conda environment and GPU 0. Logs and checkpoints are written under `runs/five_imu/`; processed data and feature caches stay under `data/`.

## Data contract

The shared MobilePoser surface synthesis produces 60 Hz wrist, shank, and waist streams in the fixed order `lw, rw, ls, rs, waist`. All AMASS and BirdDog files use the same attachment configuration hash. BirdDog was processed through the same `generate_layout_dataset` path as AMASS. Existing AMASS processed caches were checked against the same layout, rate, surface mode, and hash before use. The sixth compatibility root slot is excluded.

The encoder input is five streams of body-frame accelerometer specific force (m/s?) plus angular velocity (rad/s). Linear acceleration from the surface cache is converted to specific force using Y-up gravity, and angular velocity is computed from consecutive calibrated rotations. Each 480-frame window is encoded independently per IMU by the same frozen `imudino.pt` backbone. The five patch features are concatenated in sensor order.

The trainable probe predicts local 6D rotations for 23 SMPL articulated joints at each frame. World root orientation and root translation are outside this training target. A 10% deterministic validation split is made by sequence for AMASS and source-video SHA for BirdDog, preventing windows from one source crossing folds. BirdDog has a loss weight of 20 because it is much smaller than AMASS; its validation metric is reported separately. BirdDog pseudo-label quality still limits pose accuracy.

## Monitoring and restart

`tail -f runs/five_imu/pipeline.log` shows feature extraction and training. A finished dataset creates `data/features/five_imu_v1/<dataset>.features.pt` atomically. Rerunning the script skips finished feature caches. Training saves `latest.pt`, `best.pt`, and `metrics.jsonl` after each epoch; rerunning resumes from `latest.pt`. `runs/five_imu/COMPLETE` marks the full 30-epoch run.

The feature extraction and head fit are separate because the released IMUDINO checkpoint is frozen. The saved head requires `imudino.pt` for inference.
