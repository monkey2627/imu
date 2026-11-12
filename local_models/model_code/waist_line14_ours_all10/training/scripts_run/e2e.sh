cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5.pt MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_processed/pnp5 MOBILEPOSER_DISABLE_WANDB=1
( CUDA_VISIBLE_DEVICES=6 $P -m mobileposer.train_e2e --poser pnp5_base_poser --joints pnp5_base_joints --out pnp5_e2e --epochs 15 --lr 2e-4 > train_pnp5_e2e.log 2>&1 && CUDA_VISIBLE_DEVICES=6 $P -m mobileposer.evaluate_standard --name pnp5_e2e --poser pnp5_e2e_poser --joints pnp5_e2e_joints --mode win46 > std_pnp5_e2e.log 2>&1 ) > /dev/null 2>&1 &
( CUDA_VISIBLE_DEVICES=7 $P -m mobileposer.train_e2e --poser pnp5_base_poser --joints pnp5_base_joints --out pnp5_e2em --epochs 15 --lr 2e-4 --noise_mix 0.3 > train_pnp5_e2em.log 2>&1 && CUDA_VISIBLE_DEVICES=7 $P -m mobileposer.evaluate_standard --name pnp5_e2em --poser pnp5_e2em_poser --joints pnp5_e2em_joints --mode win46 > std_pnp5_e2em.log 2>&1 ) > /dev/null 2>&1 &
sleep 150; tail -3 train_pnp5_e2e.log | cut -c1-200
