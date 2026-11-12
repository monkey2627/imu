#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export MKL_NUM_THREADS=8
# wait for the two extra subsets
while [ ! -f data/pnp5_60_processed/pnp5/TotalCapture.pt ] || [ ! -f data/pnp5_60_processed/pnp5/EKUT.pt ]; do sleep 20; done
# paper-aligned AMASS list (PNP/PIP/TransPose): + TotalCapture, EKUT; without DanceDB, HUMAN4D
mkdir -p data/pnp5_60a/pnp5
for ds in HumanEva MPI_HDM05 SFU MPI_mosh Transitions_mocap SSM_synced CMU DFaust_67 Eyes_Japan_Dataset KIT BMLmovi EKUT TCD_handMocap ACCAD BioMotionLab_NTroje BMLhandball MPI_Limits TotalCapture; do
  ln -sf $PWD/data/pnp5_60_processed/pnp5/$ds.pt data/pnp5_60a/pnp5/$ds.pt
done
ls data/pnp5_60a/pnp5 | wc -l
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250 MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_60a/pnp5 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5_60a.pt MOBILEPOSER_DISABLE_WANDB=1 \
  $P -c "
from mobileposer.data import PoseDataset
d = PoseDataset('train'); print('windows', len(d))
" > build_cache_pnp5_60a.log 2>&1
tail -1 build_cache_pnp5_60a.log
trainmod() { # gpu module seed
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$1 MOBILEPOSER_SEED=$3 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250 MOBILEPOSER_BATCH=128 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5_60a.pt MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_60a/pnp5 MOBILEPOSER_CHECKPOINT_DIR=checkpoints/pnp5_60a_s$3_$2 MOBILEPOSER_DISABLE_WANDB=1 \
    nohup $P -m mobileposer.train --module $2 > train_pnp5_60a_s$3_$2.log 2>&1 &
}
trainmod 0 poser 42; trainmod 1 joints 42
trainmod 4 poser 7;  trainmod 5 joints 7
trainmod 6 poser 3;  trainmod 7 joints 3
echo started > chain_a.started
