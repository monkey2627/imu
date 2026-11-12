cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOBILEPOSER_FPS=60
(CUDA_VISIBLE_DEVICES=0 $P -m mobileposer.evaluate_standard --name pnp5_60a_sip3 --poser pnp5_60a_sip3_poser --joints pnp5_60a_s42_joints --mode win46 > std_pnp5_60a_sip3.log 2>&1 &)
(CUDA_VISIBLE_DEVICES=1 $P -m mobileposer.evaluate_standard --name pnp5_60a_sip6 --poser pnp5_60a_sip6_poser --joints pnp5_60a_s42_joints --mode win46 > std_pnp5_60a_sip6.log 2>&1 &)
sleep 1
