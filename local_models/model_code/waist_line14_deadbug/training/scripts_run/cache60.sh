cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
CUDA_VISIBLE_DEVICES=3 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250 MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_60_processed/pnp5 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5_60.pt MOBILEPOSER_DISABLE_WANDB=1 \
 nohup $P -c "
from mobileposer.data import PoseDataset
d = PoseDataset('train'); print('windows', len(d))
" > build_cache_pnp5_60.log 2>&1 &
sleep 2; echo started
