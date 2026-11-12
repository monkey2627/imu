cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export MOBILEPOSER_DOMAIN=1
for r in 3; do
 ( CUDA_VISIBLE_DEVICES=4 $P -m mobileposer.finetune_dip --poser pnp5_base_poser --joints pnp5_base_joints --out pnp5_dom_r$r --lr 5e-5 --epochs 20 --replay_cache data/cache_pnp5.pt --replay_ratio $r > ft_pnp5_dom_r$r.log 2>&1 && \
   CUDA_VISIBLE_DEVICES=4 $P -m mobileposer.evaluate_standard --name pnp5_dom_r$r --poser pnp5_dom_r${r}_poser --joints pnp5_dom_r${r}_joints --mode win46 > std_pnp5_dom_r$r.log 2>&1 ) &
done
sleep 1; echo started
