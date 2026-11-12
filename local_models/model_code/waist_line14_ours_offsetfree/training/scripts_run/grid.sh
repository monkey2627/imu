cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250
run() { # gpu tag lr epochs ratio
  CUDA_VISIBLE_DEVICES=$1 $P -m mobileposer.finetune_dip --poser pnp5_60a_s42_poser --joints pnp5_60a_s42_joints --out grid_$2 --lr $3 --epochs $4 --seed 42 \
    --replay_cache data/cache_pnp5_60a.pt --replay_ratio $5 --amass_val 1500 > ft_grid_$2.log 2>&1
}
gpus=(0 1 2 3 4 6 7)
jobs=("ref 0 1 0" "a_lr2e-5_e5_r0 2e-5 5 0" "a_lr2e-5_e5_r1 2e-5 5 1" "a_lr2e-5_e5_r3 2e-5 5 3" "a_lr2e-5_e5_r10 2e-5 5 10"
      "b_lr5e-5_e5_r0 5e-5 5 0" "b_lr5e-5_e5_r1 5e-5 5 1" "b_lr5e-5_e5_r3 5e-5 5 3" "b_lr5e-5_e5_r10 5e-5 5 10"
      "c_lr2e-5_e20_r0 2e-5 20 0" "c_lr2e-5_e20_r1 2e-5 20 1" "c_lr2e-5_e20_r3 2e-5 20 3" "c_lr2e-5_e20_r10 2e-5 20 10"
      "d_lr5e-5_e20_r0 5e-5 20 0" "d_lr5e-5_e20_r1 5e-5 20 1" "d_lr5e-5_e20_r3 5e-5 20 3" "d_lr5e-5_e20_r10 5e-5 20 10")
for g in 0 1 2 3 4 5 6; do
  ( i=$g; while [ $i -lt ${#jobs[@]} ]; do set -- ${jobs[$i]}; run ${gpus[$g]} $1 $2 $3 $4; i=$((i+7)); done ) &
done
wait
echo done > grid.done
