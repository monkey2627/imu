cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export MOBILEPOSER_NSENS=6
(CUDA_VISIBLE_DEVICES=0 $P -m mobileposer.evaluate_standard --name pnp6_base --poser pnp6_base_poser --joints pnp6_base_joints --mode win46 > std_pnp6_base.log 2>&1 &)
run() { # gpu name ratio
  ( CUDA_VISIBLE_DEVICES=$1 $P -m mobileposer.finetune_dip --poser pnp6_base_poser --joints pnp6_base_joints --out $2 --lr 5e-5 --epochs 20 --replay_cache data/cache_pnp6.pt --replay_ratio $3 > ft_$2.log 2>&1 && \
    CUDA_VISIBLE_DEVICES=$1 $P -m mobileposer.evaluate_standard --name $2 --poser ${2}_poser --joints ${2}_joints --mode win46 > std_$2.log 2>&1 ) &
}
run 1 pnp6_ftr3 3
run 2 pnp6_ftr10 10
( CUDA_VISIBLE_DEVICES=3 $P -m mobileposer.finetune_dip --poser pnp6_base_poser --joints pnp6_base_joints --out pnp6_ft --lr 5e-5 --epochs 20 > ft_pnp6_ft.log 2>&1 && CUDA_VISIBLE_DEVICES=3 $P -m mobileposer.evaluate_standard --name pnp6_ft --poser pnp6_ft_poser --joints pnp6_ft_joints --mode win46 > std_pnp6_ft.log 2>&1 ) &
sleep 1; echo started
