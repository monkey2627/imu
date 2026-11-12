#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/duanyuhan/dyh/motion/lines/line14
cd "$ROOT"
source /home/duanyuhan/SoftWare/miniconda3/etc/profile.d/conda.sh
conda activate mobileposer
mkdir -p results/line14/AUTO_WIN46 logs

# Gate the experiment on the completed PNP-compatible evaluation.
[[ -s results/standard/line14_win46.json ]]
OUT=results/line14/AUTO_WIN46/ori-est_geom-estp6_s42_mean.json
if [[ ! -s "$OUT" ]]; then
  exec env LINE9_GAUGE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
    python scripts_run/line14_e2e_link.py \
      --ori est --geom estp6 --lever mean --seed 42 --gpu 0 \
      --tag AUTO_WIN46 --epochs 40 --batch 128 \
      >> logs/line14_auto_validation.log 2>&1
fi
date -Is > logs/line14_auto_validation.done
