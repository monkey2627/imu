#!/usr/bin/env bash
# Process the complete five-IMU corpus, then train the downstream pose head.
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/IMUDINO-Release
PY=/home/duanyuhan/SoftWare/miniconda3/envs/imudino/bin/python
cd "$ROOT"
mkdir -p runs/five_imu data/features/five_imu_v1
exec >> runs/five_imu/pipeline.log 2>&1
trap 'echo "[pipeline] exit=$? at $(date -Is)"' EXIT

echo "[pipeline] start $(date -Is)"
echo "[pipeline] host=$(hostname) python=$PY"
CUDA_VISIBLE_DEVICES=0 "$PY" -u -m downstream.prepare --device cuda:0 --batch-size 8
echo "[pipeline] features complete $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" -u -m downstream.train --device cuda:0 --epochs 30 --batch-size 48 --bird-loss-weight 20 --output runs/five_imu
echo "[pipeline] training complete $(date -Is)"
