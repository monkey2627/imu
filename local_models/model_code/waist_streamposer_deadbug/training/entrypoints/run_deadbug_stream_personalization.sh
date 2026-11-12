#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030
WORK=$ROOT/pnp_workspace
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
mkdir -p "$ROOT/logs/personalization"

while [[ ! -f "$ROOT/logs/stream/COMPLETE" ]]; do
  if ! pgrep -f "$ROOT/run_stream_training.sh" >/dev/null; then
    echo "stream training stopped without COMPLETE" >&2
    exit 1
  fi
  sleep 30
done
cd "$WORK"
env MOBILEPOSER_FPS=60 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" "$ROOT/finetune_stream_deadbug.py" \
    --init "$ROOT/checkpoints/stream_s42_amass" \
    --store "$ROOT/deadbug60/stream_store.pt" \
    --output "$ROOT/checkpoints/stream_s42_deadbug_overfit" \
    --device cuda:6 --iters 1500 --lr 5e-5 --bs 32 --crop 1200 --seed 42 \
    >"$ROOT/logs/personalization/stream.log" 2>&1
date -Is >"$ROOT/logs/personalization/STREAM_COMPLETE"
