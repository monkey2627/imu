#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/duanyuhan/dyh/motion/lines/line14
cd "$ROOT"
source /home/duanyuhan/SoftWare/miniconda3/etc/profile.d/conda.sh
conda activate mobileposer

mkdir -p results/standard/logs
OUT=results/standard/line14_win46.json
LOG=results/standard/logs/line14_win46.log

if [[ ! -s "$OUT" ]]; then
  exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
    python scripts_run/line14_standard_eval.py --device cuda:0 --out "$OUT" \
    >> "$LOG" 2>&1
fi

# This hook is intentionally validation-only. Test metrics are never used to
# choose a candidate; a later run can enable a selected training command.
date -Is > results/standard/logs/line14_win46.done
echo "win46 result ready: $OUT"
