#!/bin/bash
# Token-Poser full pipeline: pretrain (Poser only, reuse baseline Joints) -> DIP finetune (replay x3) -> standard evaluation.
# usage: GPU=1 SEED=42 TAG=tok bash scripts_run/token_chain.sh     (run from SparseIMU/)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
SEED=${SEED:-42}; GPU=${GPU:-1}; TAG=${TAG:-tok}
NAME=pnp5_60a_s${SEED}_${TAG}
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250
export MOBILEPOSER_TOKEN_RES=${RES:-0} MOBILEPOSER_TOKEN=1 MOBILEPOSER_TOKEN_K=1024 MOBILEPOSER_TOKEN_NQ=3 MOBILEPOSER_TOKENIZER=${TOKP:-checkpoints/tokenizers/rvq3_k1024.pt}
export MOBILEPOSER_TOKEN_MODE=${MODE:-soft} MOBILEPOSER_TOKEN_SEG=${SEG:-1} MOBILEPOSER_TOKEN_W=${TW:-0.02}   # decoding mode / segment length / CE weight

if [ "${START:-1}" = 1 ]; then
# 1) pretrain the token Poser (Joints: reuse pnp5_60a_s${SEED}_joints, identical to the baseline)
MOBILEPOSER_SEED=$SEED MOBILEPOSER_BATCH=128 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5_60a.pt \
  MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_60a/pnp5 MOBILEPOSER_CHECKPOINT_DIR=checkpoints/${NAME}_poser MOBILEPOSER_DISABLE_WANDB=1 \
  $P -m mobileposer.train --module poser
fi

# 2) evaluate the pretrained model (online window setting)
$P -m mobileposer.evaluate_standard --name ${NAME} --poser ${NAME}_poser --joints pnp5_60a_s${SEED}_joints --mode win46

# 3) DIP finetune with AMASS replay x3 (same recipe as the baseline main model)
$P -m mobileposer.finetune_dip --poser ${NAME}_poser --joints pnp5_60a_s${SEED}_joints --out ${NAME}_ftr3 --lr 5e-5 --epochs 20 --train_all \
  --seed $SEED --replay_cache data/cache_pnp5_60a.pt --replay_ratio 3

# 4) evaluate the finetuned model
$P -m mobileposer.evaluate_standard --name ${NAME}_ftr3 --poser ${NAME}_ftr3_poser --joints ${NAME}_ftr3_joints --mode win46
