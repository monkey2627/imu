#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030
WORK=$ROOT/pnp_workspace
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
CACHE=$ROOT/cache/mobileposer_waist_surface_60_win250.pt
OUT=$ROOT/checkpoints/mobileposer_s42_amass
LOG=$ROOT/logs/mobileposer

common=(
  MOBILEPOSER_FPS=60
  MOBILEPOSER_WINDOW=250
  MOBILEPOSER_PROCESSED_DATASETS="$WORK/data/amass_waist_surface_60/wrists_shanks_waist"
  MOBILEPOSER_TRAIN_COMBOS=all_5imu
  MOBILEPOSER_DISABLE_WANDB=1
  WANDB_MODE=offline
  MOBILEPOSER_SEED=42
  MOBILEPOSER_DATASET_NAMES=ACCAD,BioMotionLab_NTroje,BMLhandball,BMLmovi,CMU,DFaust_67,EKUT,Eyes_Japan_Dataset,HumanEva,KIT,MPI_HDM05,MPI_Limits,MPI_mosh,SFU,SSM_synced,TCD_handMocap,TotalCapture,Transitions_mocap
  PYTHONPATH=.
)

cd "$WORK"
pids=()
for spec in poser:2 foot_contact:4 velocity:5; do
  module="${spec%%:*}"
  gpu="${spec##*:}"
  env "${common[@]}" CUDA_VISIBLE_DEVICES="$gpu" MOBILEPOSER_DATA_CACHE="$CACHE" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PY" "$ROOT/train_mobileposer_module.py" \
      --module "$module" --output "$OUT" --gpu 0 --epochs 60 --seed 42 \
      >"$LOG/$module.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more recovered MobilePoser modules failed" >&2
  exit 1
fi

while pgrep -f "train_mobileposer_module.py --module joints --output $OUT" >/dev/null; do
  sleep 30
done
for module in poser joints foot_contact velocity; do
  find "$OUT/$module" -maxdepth 1 -name '*.ckpt' -print -quit | grep -q .
done
env "${common[@]}" "$PY" "$ROOT/combine_mobileposer.py" \
  --root "$OUT" --output "$OUT/base_model.pth" >"$LOG/combine.log" 2>&1
date -Is >"$LOG/COMPLETE"
