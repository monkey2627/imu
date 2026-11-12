cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
launch() { # gpu module name cache procdir env...
  g=$1; m=$2; n=$3; c=$4; d=$5; shift 5
  env CUDA_VISIBLE_DEVICES=$g "$@" MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_$c.pt MOBILEPOSER_PROCESSED_DATASETS=$d MOBILEPOSER_CHECKPOINT_DIR=checkpoints/${n}_$m MOBILEPOSER_DISABLE_WANDB=1 \
    nohup $P -m mobileposer.train --module $m > train_${n}_$m.log 2>&1 &
}
launch 0 poser  pnp5_60 pnp5_60 data/pnp5_60_processed/pnp5 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250 MOBILEPOSER_BATCH=128
launch 1 joints pnp5_60 pnp5_60 data/pnp5_60_processed/pnp5 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250 MOBILEPOSER_BATCH=128
launch 2 poser  pnp5_pelvis pnp5_pelvis data/pnp5_processed/pnp5 MOBILEPOSER_FRAME=pelvis
launch 3 joints pnp5_pelvis pnp5_pelvis data/pnp5_processed/pnp5 MOBILEPOSER_FRAME=pelvis
launch 6 poser  pnp5_yaw pnp5_yaw data/pnp5_processed/pnp5 MOBILEPOSER_FRAME=yaw
launch 7 joints pnp5_yaw pnp5_yaw data/pnp5_processed/pnp5 MOBILEPOSER_FRAME=yaw
sleep 90
for n in pnp5_60_poser pnp5_pelvis_poser pnp5_yaw_poser; do tail -c 250 train_$n.log | tr "\r" "\n" | tail -1 | cut -c1-120; done
