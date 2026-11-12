#!/bin/bash
# Environment-B measurement (line 14): R1 oracle vs B0/B1, 60 epochs, shape-aware labels, validation-only.
# Jobs: oracle x seeds {42,7,3}, b1 x seeds {42,7,3}, b0 seed 42 (reproducibility check vs line9 shape/b0_s42).
cd /home/duanyuhan/dyh/motion/lines/line14
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$PWD
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
run() { mode=$1; seed=$2; gpu=$3; nohup $PY scripts_run/line14_envB.py --mode $mode --seed $seed --gpu $gpu > logs/line14/envB_${mode}_s${seed}.log 2>&1 & echo "$mode s$seed gpu$gpu pid $!"; }
run oracle 42 0; run oracle 7 1; run oracle 3 2
run b1 42 3; run b1 7 4; run b1 3 6
run b0 42 7
