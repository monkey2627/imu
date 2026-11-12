#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030
WORK=$ROOT/pnp_workspace
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
DATA=$WORK/data/amass_waist_surface_60/wrists_shanks_waist
CURATED=$WORK/data/amass_waist_surface_60_paper18
STORE=$ROOT/data/stream_waist_surface_60.pt
OUT=stream_waist_surface_both_mount8_s42_amass
LOG=$ROOT/logs/stream
mkdir -p "$ROOT/data" "$LOG"

while [[ ! -f "$ROOT/logs/preprocess/COMPLETE" ]]; do
  if ! pgrep -f "$ROOT/run_preprocess.sh" >/dev/null; then
    echo "preprocess stopped without COMPLETE" >&2
    exit 1
  fi
  sleep 30
done

cd "$WORK"
mkdir -p "$CURATED"
for name in ACCAD BioMotionLab_NTroje BMLhandball BMLmovi CMU DFaust_67 EKUT Eyes_Japan_Dataset HumanEva KIT MPI_HDM05 MPI_Limits MPI_mosh SFU SSM_synced TCD_handMocap TotalCapture Transitions_mocap; do
  ln -sfn "$DATA/$name.pt" "$CURATED/$name.pt"
done
env CUDA_VISIBLE_DEVICES=6 MOBILEPOSER_FPS=60 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" -m mobileposer.stream.data --amass "$CURATED" --out "$STORE" >"$LOG/store.log" 2>&1
env CUDA_VISIBLE_DEVICES=6 MOBILEPOSER_FPS=60 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" -m mobileposer.stream.train --out "$OUT" --amass "$STORE" \
    --iters 20000 --bs 32 --crop 1200 --lag 5 --lr 1e-3 --wj 10 \
    --gyro 1 --mount_deg 8 --frame both --seed 42 >"$LOG/train.log" 2>&1
mkdir -p "$ROOT/checkpoints/stream_s42_amass"
cp -a "$WORK/checkpoints_research/$OUT/." "$ROOT/checkpoints/stream_s42_amass/"
date -Is >"$LOG/COMPLETE"
