#!/usr/bin/env bash
# Synthesize wrist/ankle five-IMU surface data (AMASS paper-18 + BirdDog), 60 Hz.
set -euo pipefail
N=/home/duanyuhan/dyh/motion/lines/imudino_wristankle_20261112
WS=$N/synth_ws
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
OUT=$N/data/processed
CFG=$WS/mobileposer/configs/wristankle_surface_imu.json
mkdir -p "$OUT" "$N/logs/synth"
groups=(
  "BioMotionLab_NTroje"
  "CMU ACCAD"
  "BMLmovi BMLhandball EKUT"
  "KIT MPI_HDM05 HumanEva DFaust_67"
  "Eyes_Japan_Dataset MPI_Limits MPI_mosh SFU SSM_synced TCD_handMocap TotalCapture Transitions_mocap BirdDog"
)
cd "$WS"
pids=()
for i in "${!groups[@]}"; do
  read -r -a ds <<< "${groups[$i]}"
  env MOBILEPOSER_FPS=60 MOBILEPOSER_RESAMPLE_ANY_FPS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. \
    "$PY" -m mobileposer.no_head_layouts --layout wrists_shanks_waist --output-root "$OUT" \
      --source amass --imu-mode surface --attachment-config "$CFG" --datasets "${ds[@]}" \
      --mesh-chunk-size 128 >"$N/logs/synth/group_$i.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || { echo "synthesis failed" >&2; exit 1; }
for f in "$OUT"/wrists_shanks_waist/*.pt; do PYTHONPATH=. "$PY" "$N/validate_surface_data.py" "$f" "$CFG"; done > "$N/logs/synth/validation.log" 2>&1
date -Is > "$N/logs/synth/COMPLETE"
