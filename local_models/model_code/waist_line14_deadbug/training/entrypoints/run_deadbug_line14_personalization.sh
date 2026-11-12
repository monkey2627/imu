#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030
WORK=$ROOT/line14_workspace
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
mkdir -p "$ROOT/logs/personalization" "$ROOT/checkpoints/line14_s42_deadbug_overfit"

while [[ ! -f "$ROOT/logs/line14/COMPLETE" ]]; do
  if ! pgrep -f "$ROOT/run_line14_training.sh" >/dev/null; then
    echo "line14 training stopped without COMPLETE" >&2
    exit 1
  fi
  sleep 30
done
cd "$WORK"
env MOBILEPOSER_FPS=60 PYTHONPATH=.:scripts_run OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" "$ROOT/finetune_line14_deadbug.py" \
    --checkpoint "$ROOT/checkpoints/line14_s42_amass/ori-none_geom-none_s42.pt" \
    --input "$ROOT/deadbug60/inputs/wrists_shanks_waist/ours1.pt" \
    --targets "$ROOT/deadbug60/targets_fbx_smpl_60.pt" \
    --output "$ROOT/checkpoints/line14_s42_deadbug_overfit/model.pt" \
    --device cuda:7 --steps 500 --crop 600 --lr 1e-4 --seed 42 \
    >"$ROOT/logs/personalization/line14.log" 2>&1
date -Is >"$ROOT/logs/personalization/LINE14_COMPLETE"
