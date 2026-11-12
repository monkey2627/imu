#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030
WORK=$ROOT/pnp_workspace
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
INPUT=$ROOT/deadbug60/inputs/wrists_shanks_waist/ours1.pt
TARGET=$ROOT/deadbug60/targets_fbx_smpl_60.pt
STORE=$ROOT/deadbug60/stream_store.pt
mkdir -p "$ROOT/logs/personalization"

while [[ ! -f "$ROOT/logs/mobileposer/COMPLETE" ]]; do
  if ! pgrep -f "$ROOT/run_pnp_training.sh" >/dev/null \
      && ! pgrep -f "$ROOT/recover_pnp_training.sh" >/dev/null; then
    echo "MobilePoser training stopped without COMPLETE" >&2
    exit 1
  fi
  sleep 30
done
cd "$WORK"
env MOBILEPOSER_FPS=60 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" "$ROOT/finetune_mobileposer_deadbug.py" \
    --model "$ROOT/checkpoints/mobileposer_s42_amass/base_model.pth" \
    --input "$INPUT" --targets "$TARGET" \
    --output "$ROOT/checkpoints/mobileposer_s42_deadbug_overfit" \
    --device cuda:2 --steps 500 --crop 600 --lr 1e-4 --seed 42 \
    >"$ROOT/logs/personalization/mobileposer.log" 2>&1
date -Is >"$ROOT/logs/personalization/MOBILEPOSER_COMPLETE"
