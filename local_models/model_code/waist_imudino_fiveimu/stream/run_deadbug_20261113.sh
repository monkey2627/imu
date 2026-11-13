#!/usr/bin/env bash
set -euo pipefail
N=/home/duanyuhan/dyh/motion/lines/imudino_deadbug_20261113
BASE=/home/duanyuhan/dyh/motion/lines/imudino_wristankle_20261112
PY=/home/duanyuhan/SoftWare/miniconda3/envs/imudino/bin/python
SYNTHPY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
CFG=$BASE/synth_ws/mobileposer/configs/wristankle_surface_imu.json
SRC=$N/data/processed/wrists_shanks_waist
FEAT=$N/data/features/stream_wristankle_v1
export CUDA_VISIBLE_DEVICES=${DEADBUG_GPU:-0}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$SRC" "$FEAT" "$N/logs"
trap 'echo "FAILED line=$LINENO" > "$N/logs/FAILED"' ERR
if [[ ! -f "$SRC/split_report.json" ]]; then
  MOBILEPOSER_FPS=60 MOBILEPOSER_RESAMPLE_ANY_FPS=1 PYTHONPATH="$BASE/synth_ws:$BASE/imudino_stream" \
    "$SYNTHPY" -u "$N/synth_deadbug.py" \
      --raw /home/duanyuhan/dyh/motion/data_system/runtime/output/training_data/DeadBug \
      --out "$SRC" --config "$CFG" > "$N/logs/synth.log" 2>&1
fi
for dataset in DeadBug DeadBugMix; do
  "$PY" "$BASE/validate_surface_data.py" "$SRC/$dataset.pt" "$CFG" > "$N/logs/validate_$dataset.json"
done
datasets=(ACCAD BioMotionLab_NTroje BMLhandball BMLmovi CMU DFaust_67 EKUT Eyes_Japan_Dataset HumanEva KIT
          MPI_HDM05 MPI_Limits MPI_mosh SFU SSM_synced TCD_handMocap TotalCapture Transitions_mocap)
for dataset in "${datasets[@]}"; do
  test -f "$BASE/data/features/stream_wristankle_v1/$dataset.stream.pt"
  if [[ ! -e "$FEAT/$dataset.stream.pt" ]]; then
    ln -s "$BASE/data/features/stream_wristankle_v1/$dataset.stream.pt" "$FEAT/$dataset.stream.pt"
  fi
done
cd "$BASE/imudino_stream"
"$PY" -u -m stream.prepare_stream --source-root "$SRC" --datasets DeadBug DeadBugMix --out "$FEAT" \
  --attachment-sha256 0e9da789a72715b2cc165bb65fc0ac285b5a8334ff1b8f6f3e25fc9adce1c338 --device cuda:0 \
  > "$N/logs/prepare.log" 2>&1
date -Is > "$N/logs/PREPARED"
"$PY" -u -m stream.train_stream --cache-root "$FEAT" --datasets "${datasets[@]}" DeadBug DeadBugMix \
  --special-prefix DeadBug --special-name deadbug --output "$N/runs/stream_deadbug_limbmix" \
  --steps 60000 --batch 64 --val-every 2000 --bird-fraction 0.10 > "$N/logs/train.log" 2>&1
date -Is > "$N/logs/COMPLETE"
