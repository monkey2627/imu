cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_FPS=60 MOBILEPOSER_WINDOW=250 MOBILEPOSER_SHORTCUT=1
while ! tr "\r" "\n" < train_pnp5_60a_sc_poser.log | grep -q "Epoch 59: 100%"; do sleep 20; done; sleep 30
CUDA_VISIBLE_DEVICES=2 $P -m mobileposer.evaluate_standard --name pnp5_60a_sc --poser pnp5_60a_sc_poser --joints pnp5_60a_s42_joints --mode win46 > std_pnp5_60a_sc.log 2>&1 &
for r in 3 1; do
  g=$([ $r = 3 ] && echo 3 || echo 4)
  ( CUDA_VISIBLE_DEVICES=$g $P -m mobileposer.finetune_dip --poser pnp5_60a_sc_poser --joints pnp5_60a_s42_joints --out pnp5_60a_sc_ftr$r --lr 5e-5 --epochs 20 --train_all --seed 42 --replay_cache data/cache_pnp5_60a.pt --replay_ratio $r > ft_pnp5_60a_sc_ftr$r.log 2>&1 \
    && CUDA_VISIBLE_DEVICES=$g $P -m mobileposer.evaluate_standard --name pnp5_60a_sc_ftr$r --poser pnp5_60a_sc_ftr${r}_poser --joints pnp5_60a_sc_ftr${r}_joints --mode win46 > std_pnp5_60a_sc_ftr$r.log 2>&1 ) &
done
wait
echo done > sc_chain.done
