#!/bin/bash
# Joints-side variants (bone-chain output / privileged distillation): train Joints only, pair with the baseline Poser, then DIP finetune.
# usage: GPU=2 SEED=42 TAG=jchain JCHAIN=1 [PRIV=1] bash scripts_run/joints_chain.sh     (run from SparseIMU/)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
SEED=${SEED:-42}; GPU=${GPU:-1}; TAG=${TAG:-jvar}
NAME=pnp5_60a_s${SEED}_${TAG}
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250
export MOBILEPOSER_JCHAIN=${JCHAIN:-0} MOBILEPOSER_PRIV=${PRIV:-0}
POSER=pnp5_60a_s${SEED}_poser                                   # baseline Poser, unchanged

if [ "${START:-1}" = 1 ]; then
MOBILEPOSER_SEED=$SEED MOBILEPOSER_BATCH=128 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5_60a.pt \
  MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_60a/pnp5 MOBILEPOSER_CHECKPOINT_DIR=checkpoints/${NAME}_joints MOBILEPOSER_DISABLE_WANDB=1 \
  $P -m mobileposer.train --module joints
fi

$P -m mobileposer.evaluate_standard --name ${NAME} --poser ${POSER} --joints ${NAME}_joints --mode win46

export MOBILEPOSER_PRIV_DISTILL=0                               # real DIP data has no privileged sensors: student-only fine-tuning
$P -m mobileposer.finetune_dip --poser ${POSER} --joints ${NAME}_joints --out ${NAME}_ftr3 --lr 5e-5 --epochs 20 --train_all \
  --seed $SEED --replay_cache data/cache_pnp5_60a.pt --replay_ratio 3

$P -m mobileposer.evaluate_standard --name ${NAME}_ftr3 --poser ${NAME}_ftr3_poser --joints ${NAME}_ftr3_joints --mode win46
