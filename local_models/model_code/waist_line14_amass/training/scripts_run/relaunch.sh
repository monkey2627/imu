cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
pkill -f "mobileposer.data import PoseDataset" ; pkill -f "finetune_dip --poser pnp5_base_poser --joints pnp5_base_joints --out pnp5_dom_r3"; sleep 3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
build() { # gpu name env...
  g=$1; n=$2; shift 2
  env CUDA_VISIBLE_DEVICES=$g "$@" MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_$n.pt MOBILEPOSER_DISABLE_WANDB=1 \
    nohup $P -c "
from mobileposer.data import PoseDataset
d = PoseDataset('train'); print('windows', len(d))
" > build_cache_$n.log 2>&1 &
}
rm -f data/cache_pnp5_60.pt.tmp data/cache_pnp5_pelvis.pt.tmp data/cache_pnp5_yaw.pt.tmp
build 3 pnp5_60 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250 MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_60_processed/pnp5
build 5 pnp5_pelvis MOBILEPOSER_FRAME=pelvis MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_processed/pnp5
build 6 pnp5_yaw MOBILEPOSER_FRAME=yaw MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_processed/pnp5
( export OMP_NUM_THREADS=8 MOBILEPOSER_DOMAIN=1; CUDA_VISIBLE_DEVICES=4 nohup $P -m mobileposer.finetune_dip --poser pnp5_base_poser --joints pnp5_base_joints --out pnp5_dom_r3 --lr 5e-5 --epochs 20 --replay_cache data/cache_pnp5.pt --replay_ratio 3 > ft_pnp5_dom_r3.log 2>&1 && CUDA_VISIBLE_DEVICES=4 nohup $P -m mobileposer.evaluate_standard --name pnp5_dom_r3 --poser pnp5_dom_r3_poser --joints pnp5_dom_r3_joints --mode win46 > std_pnp5_dom_r3.log 2>&1 ) > /dev/null 2>&1 &
sleep 120
for n in pnp5_60 pnp5_pelvis pnp5_yaw; do tail -c 130 build_cache_$n.log | tr "\r" "\n" | tail -1; done
