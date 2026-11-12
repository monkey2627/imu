cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
for k in pelvis yaw; do
  CUDA_VISIBLE_DEVICES=7 MOBILEPOSER_FRAME=$k MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_processed/pnp5 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5_$k.pt MOBILEPOSER_DISABLE_WANDB=1 \
   nohup $P -c "
from mobileposer.data import PoseDataset
d = PoseDataset('train'); print('windows', len(d))
" > build_cache_pnp5_$k.log 2>&1 &
done
sleep 60; tail -c 150 build_cache_pnp5_yaw.log | tr "\r" "\n" | tail -1
