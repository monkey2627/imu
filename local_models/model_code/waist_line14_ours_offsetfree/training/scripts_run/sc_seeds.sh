cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250 MOBILEPOSER_SHORTCUT=1
seedrun() { # gpu seed
  g=$1; s=$2
  ( MOBILEPOSER_SEED=$s MOBILEPOSER_BATCH=128 MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DATA_CACHE=data/cache_pnp5_60a.pt MOBILEPOSER_PROCESSED_DATASETS=data/pnp5_60a/pnp5 MOBILEPOSER_CHECKPOINT_DIR=checkpoints/pnp5_60a_s${s}sc_poser MOBILEPOSER_DISABLE_WANDB=1 CUDA_VISIBLE_DEVICES=$g \
      $P -m mobileposer.train --module poser > train_pnp5_60a_s${s}sc_poser.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $P -m mobileposer.evaluate_standard --name pnp5_60a_s${s}sc --poser pnp5_60a_s${s}sc_poser --joints pnp5_60a_s${s}_joints --mode win46 > std_pnp5_60a_s${s}sc.log 2>&1
    for r in 3 1; do
      CUDA_VISIBLE_DEVICES=$g $P -m mobileposer.finetune_dip --poser pnp5_60a_s${s}sc_poser --joints pnp5_60a_s${s}_joints --out pnp5_60a_s${s}sc_ftr$r --lr 5e-5 --epochs 20 --train_all --seed $s --replay_cache data/cache_pnp5_60a.pt --replay_ratio $r > ft_pnp5_60a_s${s}sc_ftr$r.log 2>&1 \
       && CUDA_VISIBLE_DEVICES=$g $P -m mobileposer.evaluate_standard --name pnp5_60a_s${s}sc_ftr$r --poser pnp5_60a_s${s}sc_ftr${r}_poser --joints pnp5_60a_s${s}sc_ftr${r}_joints --mode win46 > std_pnp5_60a_s${s}sc_ftr$r.log 2>&1
    done ) &
}
seedrun 0 7
seedrun 1 3
wait
echo done > sc_seeds.done
