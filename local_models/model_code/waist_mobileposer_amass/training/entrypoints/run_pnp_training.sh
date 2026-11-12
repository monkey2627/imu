#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030
WORK=$ROOT/pnp_workspace
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
DATA=$WORK/data/amass_waist_surface_60/wrists_shanks_waist
CACHE=$ROOT/cache/mobileposer_waist_surface_60_win250.pt
OUT=$ROOT/checkpoints/mobileposer_s42_amass
LOG=$ROOT/logs/mobileposer
mkdir -p "$ROOT/cache" "$OUT" "$LOG"

while [[ ! -f "$ROOT/logs/preprocess/COMPLETE" ]]; do
  if ! pgrep -f "$ROOT/run_preprocess.sh" >/dev/null; then
    echo "preprocess stopped without COMPLETE" >&2
    exit 1
  fi
  sleep 30
done

common=(
  MOBILEPOSER_FPS=60
  MOBILEPOSER_WINDOW=250
  MOBILEPOSER_PROCESSED_DATASETS="$DATA"
  MOBILEPOSER_TRAIN_COMBOS=all_5imu
  MOBILEPOSER_DISABLE_WANDB=1
  WANDB_MODE=offline
  MOBILEPOSER_SEED=42
  MOBILEPOSER_DATASET_NAMES=ACCAD,BioMotionLab_NTroje,BMLhandball,BMLmovi,CMU,DFaust_67,EKUT,Eyes_Japan_Dataset,HumanEva,KIT,MPI_HDM05,MPI_Limits,MPI_mosh,SFU,SSM_synced,TCD_handMocap,TotalCapture,Transitions_mocap
  PYTHONPATH=.
)

cd "$WORK"
env "${common[@]}" MOBILEPOSER_DATA_CACHE="$CACHE" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" "$ROOT/build_mobileposer_cache.py" >"$LOG/cache.log" 2>&1

pids=()
modules=(poser joints foot_contact velocity)
for i in "${!modules[@]}"; do
  module=${modules[$i]}
  gpu="$((i + 2))"
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
  echo "one or more MobilePoser modules failed" >&2
  exit 1
fi

env "${common[@]}" "$PY" "$ROOT/combine_mobileposer.py" \
  --root "$OUT" --output "$OUT/base_model.pth" >"$LOG/combine.log" 2>&1
date -Is >"$LOG/COMPLETE"
