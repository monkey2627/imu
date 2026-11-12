cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for w in 3 6; do
  g=$([ $w = 3 ] && echo 0 || echo 1)
  CUDA_VISIBLE_DEVICES=$g MOBILEPOSER_SIPW=$w MOBILEPOSER_SEED=42 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250 MOBILEPOSER_BATCH=128 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5_60a.pt MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_60a/pnp5 MOBILEPOSER_CHECKPOINT_DIR=checkpoints/pnp5_60a_sip${w}_poser MOBILEPOSER_DISABLE_WANDB=1 \
    setsid nohup $P -m mobileposer.train --module poser > train_pnp5_60a_sip${w}_poser.log 2>&1 < /dev/null &
done
sleep 60; for w in 3 6; do tail -c 160 train_pnp5_60a_sip${w}_poser.log | tr "\r" "\n" | tail -1 | cut -c1-100; done
