#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030
WORK=$ROOT/line14_workspace
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
DATA=$ROOT/pnp_workspace/data/amass_waist_surface_60/wrists_shanks_waist
AUDIT=$WORK/results/line9/data_audit.json
SHAPE=$WORK/data/line9_store_shape
POSE=$WORK/data/line9_store_pose6d
LOG=$ROOT/logs/line14
mkdir -p "$WORK/data/pnp5_60a" "$WORK/results/line9" "$LOG"

if [[ ! -f "$WORK/third_party/PNP/articulate/__init__.py" ]]; then
  cp -a /home/duanyuhan/dyh/motion/lines/research_20261027/baselines/b0_line14/third_party "$WORK/third_party"
fi
mkdir -p "$WORK/data_local"
if [[ ! -f "$WORK/data_local/line14_est_r2_b90.pt" ]]; then
  cp /home/duanyuhan/dyh/motion/lines/line14/data_local/line14_est_r2_b90.pt "$WORK/data_local/"
fi

while [[ ! -f "$ROOT/logs/preprocess/COMPLETE" ]]; do
  if ! pgrep -f "$ROOT/run_preprocess.sh" >/dev/null; then
    echo "preprocess stopped without COMPLETE" >&2
    exit 1
  fi
  sleep 30
done

ln -sfn "$DATA" "$WORK/data/pnp5_60a/pnp5"
cp /home/duanyuhan/dyh/motion/lines/line14/results/line9/data_audit.json "$AUDIT"
cd "$WORK"
if [[ ! -f "$SHAPE/index.json" || ! -f "$POSE/index.json" ]]; then
  env MOBILEPOSER_FPS=60 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PY" "$ROOT/build_line14_stores.py" --data "$DATA" --audit "$AUDIT" \
      --shape-store "$SHAPE" --pose-store "$POSE" --workers 12 >"$LOG/store.log" 2>&1
fi

env LINE9_GAUGE=1 MOBILEPOSER_FPS=60 PYTHONPATH=.:scripts_run OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" scripts_run/line14_e2e_link.py --ori none --geom none --seed 42 --gpu 7 \
    --tag waist_surface_smoke --epochs 1 --imu-store "$SHAPE" --rot-deg 0 --lever-cm 0 --smoke \
    >"$LOG/smoke.log" 2>&1

env LINE9_GAUGE=1 MOBILEPOSER_FPS=60 PYTHONPATH=.:scripts_run OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" scripts_run/line14_e2e_link.py --ori none --geom none --seed 42 --gpu 7 \
    --tag waist_surface_s42_amass --epochs 40 --imu-store "$SHAPE" --rot-deg 0 --lever-cm 0 \
    --scale-range 0.5,1.2 >"$LOG/train.log" 2>&1

mkdir -p "$ROOT/checkpoints/line14_s42_amass"
cp -a "$WORK/checkpoints/line14/waist_surface_s42_amass/." "$ROOT/checkpoints/line14_s42_amass/"
date -Is >"$LOG/COMPLETE"
