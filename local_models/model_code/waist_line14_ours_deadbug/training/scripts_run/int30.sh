cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
launch() { g=$1; m=$2; n=$3; shift 3
  env CUDA_VISIBLE_DEVICES=$g "$@" MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5.pt MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_processed/pnp5 MOBILEPOSER_CHECKPOINT_DIR=checkpoints/${n}_$m MOBILEPOSER_DISABLE_WANDB=1 \
    setsid nohup $P -m mobileposer.train --module $m > train_${n}_$m.log 2>&1 < /dev/null &
}
launch 2 poser  pnp5_int MOBILEPOSER_INTEG=1
launch 3 joints pnp5_int MOBILEPOSER_INTEG=1
sleep 90
for n in pnp5_int_poser pnp5_int_joints; do tail -c 200 train_$n.log | tr "\r" "\n" | tail -1 | cut -c1-110; done
