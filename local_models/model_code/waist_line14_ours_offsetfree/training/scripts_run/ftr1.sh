cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250
one() { # gpu seed
  CUDA_VISIBLE_DEVICES=$1 $P -m mobileposer.finetune_dip --poser pnp5_60a_s$2_poser --joints pnp5_60a_s$2_joints --out pnp5_60a_s$2_ftr1 --lr 5e-5 --epochs 20 --train_all --seed $2 --replay_cache data/cache_pnp5_60a.pt --replay_ratio 1 > ft_pnp5_60a_s$2_ftr1.log 2>&1 \
   && CUDA_VISIBLE_DEVICES=$1 $P -m mobileposer.evaluate_standard --name pnp5_60a_s$2_ftr1 --poser pnp5_60a_s$2_ftr1_poser --joints pnp5_60a_s$2_ftr1_joints --mode win46 > std_pnp5_60a_s$2_ftr1.log 2>&1
}
one 0 42 & one 4 7 & one 7 3 &
wait
echo done > ftr1.done
