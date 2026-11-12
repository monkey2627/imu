cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
launch() { # gpu module name env...
  g=$1; m=$2; n=$3; shift 3
  env CUDA_VISIBLE_DEVICES=$g "$@" MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5.pt MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_processed/pnp5 MOBILEPOSER_CHECKPOINT_DIR=checkpoints/${n}_$m MOBILEPOSER_DISABLE_WANDB=1 \
    nohup $P -m mobileposer.train --module $m > train_${n}_$m.log 2>&1 &
}
launch 0 poser  pnp5_cos MOBILEPOSER_SCHED=cosine
launch 1 joints pnp5_cos MOBILEPOSER_SCHED=cosine
launch 2 poser  pnp5_sc MOBILEPOSER_SHORTCUT=1
launch 3 poser  pnp5_cossc MOBILEPOSER_SCHED=cosine MOBILEPOSER_SHORTCUT=1
sleep 60
for n in pnp5_cos_poser pnp5_sc_poser pnp5_cossc_poser; do tail -c 200 train_$n.log | tr "\r" "\n" | tail -1 | cut -c1-110; done
