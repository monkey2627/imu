cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250
ev() { # gpu name poser joints
  CUDA_VISIBLE_DEVICES=$1 $P -m mobileposer.evaluate_standard --name $2 --poser $3 --joints $4 --mode win46 > std_$2.log 2>&1
}
ft() { # gpu seed tag extra...
  g=$1; s=$2; tag=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g $P -m mobileposer.finetune_dip --poser pnp5_60a_s${s}_poser --joints pnp5_60a_s${s}_joints --out pnp5_60a_s${s}_$tag --lr 5e-5 --epochs 20 --train_all --seed $s "$@" > ft_pnp5_60a_s${s}_$tag.log 2>&1 \
   && ev $g pnp5_60a_s${s}_$tag pnp5_60a_s${s}_${tag}_poser pnp5_60a_s${s}_${tag}_joints
}
seed() { # gpuA gpuB seed
  ( ev $1 pnp5_60a_s$3 pnp5_60a_s$3_poser pnp5_60a_s$3_joints; ft $1 $3 ft ) &
  ( ft $2 $3 ftr3 --replay_cache data/cache_pnp5_60a.pt --replay_ratio 3 ) &
}
seed 0 1 42
seed 4 6 7
seed 7 5 3
wait
echo done > aligned_eval.done
