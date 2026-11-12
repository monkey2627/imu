cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
(CUDA_VISIBLE_DEVICES=2 $P -m mobileposer.evaluate_standard --name pnp5_cos --poser pnp5_cos_poser --joints pnp5_cos_joints --mode win46 > std_pnp5_cos.log 2>&1 &)
(CUDA_VISIBLE_DEVICES=3 MOBILEPOSER_SHORTCUT=1 $P -m mobileposer.evaluate_standard --name pnp5_sc --poser pnp5_sc_poser --joints pnp5_base_joints --mode win46 > std_pnp5_sc.log 2>&1 &)
(CUDA_VISIBLE_DEVICES=2 MOBILEPOSER_SHORTCUT=1 $P -m mobileposer.evaluate_standard --name pnp5_cossc --poser pnp5_cossc_poser --joints pnp5_cos_joints --mode win46 > std_pnp5_cossc.log 2>&1 &)
sleep 1
