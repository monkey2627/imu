cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
run() { # gpu name ratio
  ( CUDA_VISIBLE_DEVICES=$1 $P -m mobileposer.finetune_dip --poser pnp5_base_poser --joints pnp5_base_joints --out $2 --lr 5e-5 --epochs 20 --replay_cache data/cache_pnp5.pt --replay_ratio $3 > ft_$2.log 2>&1 && \
    CUDA_VISIBLE_DEVICES=$1 $P -m mobileposer.evaluate_standard --name $2 --poser ${2}_poser --joints ${2}_joints --mode win46 > std_$2.log 2>&1 ) &
}
run 0 pnp5_ftr3 3
run 1 pnp5_ftr10 10
sleep 1; echo started
