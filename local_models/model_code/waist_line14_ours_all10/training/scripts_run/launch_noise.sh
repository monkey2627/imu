cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
D=data/pnp5_processed/pnp5
launch() { # gpu module name ori jit acc bias gain
  CUDA_VISIBLE_DEVICES=$1 MOBILEPOSER_NOISE=1 MOBILEPOSER_NOISE_ORI_DEG=$4 MOBILEPOSER_NOISE_JIT_DEG=$5 MOBILEPOSER_NOISE_ACC=$6 MOBILEPOSER_NOISE_BIAS=$7 MOBILEPOSER_NOISE_GAIN=$8 \
  MOBILEPOSER_DATA_CACHE=data/cache_pnp5.pt MOBILEPOSER_PROCESSED_DATASETS=$D MOBILEPOSER_CHECKPOINT_DIR=checkpoints/$3 \
  MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DISABLE_WANDB=1 nohup $P -m mobileposer.train --module $2 > train_$3.log 2>&1 &
}
launch 0 poser  pnp5_n1_poser  5 1 0.3 0.2 0.1
launch 1 joints pnp5_n1_joints 5 1 0.3 0.2 0.1
launch 2 poser  pnp5_n2_poser  10 2 0.6 0.4 0.2
launch 3 joints pnp5_n2_joints 10 2 0.6 0.4 0.2
sleep 3; echo launched
