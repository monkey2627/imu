#!/bin/bash
# Anchoring variants (shortcut + gate / kinematic-relative parents / sensor-mounting-error augmentation): pretrain Poser (reuse baseline Joints)
# -> evaluate -> DIP finetune (replay x3) -> evaluate.
# usage: GPU=2 SEED=42 TAG=anchgate ANCHOR=gate BIAS=15 bash scripts_run/anchor_chain.sh     (run from SparseIMU/)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
SEED=${SEED:-42}; GPU=${GPU:-1}; TAG=${TAG:-anch}
NAME=pnp5_60a_s${SEED}_${TAG}
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250
export MOBILEPOSER_SHORTCUT=1 MOBILEPOSER_ANCHOR=${ANCHOR:-} MOBILEPOSER_BIAS_DEG=${BIAS:-0} MOBILEPOSER_BIAS_P=${BIAS_P:-0.5}

if [ "${START:-1}" = 1 ]; then
MOBILEPOSER_SEED=$SEED MOBILEPOSER_BATCH=128 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5_60a.pt \
  MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_60a/pnp5 MOBILEPOSER_CHECKPOINT_DIR=checkpoints/${NAME}_poser MOBILEPOSER_DISABLE_WANDB=1 \
  $P -m mobileposer.train --module poser
fi

$P -m mobileposer.evaluate_standard --name ${NAME} --poser ${NAME}_poser --joints pnp5_60a_s${SEED}_joints --mode win46

$P -m mobileposer.finetune_dip --poser ${NAME}_poser --joints pnp5_60a_s${SEED}_joints --out ${NAME}_ftr3 --lr 5e-5 --epochs 20 --train_all \
  --seed $SEED --replay_cache data/cache_pnp5_60a.pt --replay_ratio 3

$P -m mobileposer.evaluate_standard --name ${NAME}_ftr3 --poser ${NAME}_ftr3_poser --joints ${NAME}_ftr3_joints --mode win46
