#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030
WORK=$ROOT/pnp_workspace
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
OUT=$WORK/data/amass_waist_surface_60
CFG=$WORK/mobileposer/configs/no_head_surface_imu.json
mkdir -p "$OUT" "$ROOT/logs/preprocess"

groups=(
  "ACCAD BioMotionLab_NTroje BMLhandball BMLmovi CMU"
  "DanceDB DFaust_67 EKUT Eyes_Japan_Dataset HUMAN4D"
  "HumanEva KIT MPI_HDM05 MPI_Limits MPI_mosh"
  "SFU SSM_synced TCD_handMocap TotalCapture Transitions_mocap"
)

pids=()
cd "$WORK"
for i in "${!groups[@]}"; do
  read -r -a datasets <<< "${groups[$i]}"
  env MOBILEPOSER_FPS=60 MOBILEPOSER_RESAMPLE_ANY_FPS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
    "$PY" -m mobileposer.no_head_layouts \
      --layout wrists_shanks_waist \
      --output-root "$OUT" \
      --source amass \
      --imu-mode surface \
      --attachment-config "$CFG" \
      --datasets "${datasets[@]}" \
      --mesh-chunk-size 128 \
      --overwrite \
      >"$ROOT/logs/preprocess/group_$i.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more preprocess workers failed" >&2
  exit 1
fi

find "$OUT/wrists_shanks_waist" -maxdepth 1 -type f -name '*.pt' -print0 \
  | sort -z \
  | while IFS= read -r -d '' file; do
      PYTHONPATH=. "$PY" "$ROOT/validate_surface_data.py" "$file" "$CFG"
    done >"$ROOT/logs/preprocess/validation.log" 2>&1

date -Is >"$ROOT/logs/preprocess/COMPLETE"
