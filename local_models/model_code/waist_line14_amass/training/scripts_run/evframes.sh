cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
ls checkpoints/pnp5_pelvis_poser/*/poser | tail -2; ls checkpoints/pnp5_yaw_poser/*/poser | tail -2
tail -c 200 train_pnp5_pelvis_poser.log | tr "\r" "\n" | tail -1 | cut -c1-80; tail -c 200 train_pnp5_yaw_poser.log | tr "\r" "\n" | tail -1 | cut -c1-80
(CUDA_VISIBLE_DEVICES=2 MOBILEPOSER_FRAME=pelvis $P -m mobileposer.evaluate_standard --name pnp5_pelvis --poser pnp5_pelvis_poser --joints pnp5_pelvis_joints --mode win46 > std_pnp5_pelvis.log 2>&1 &)
(CUDA_VISIBLE_DEVICES=3 MOBILEPOSER_FRAME=yaw $P -m mobileposer.evaluate_standard --name pnp5_yaw --poser pnp5_yaw_poser --joints pnp5_yaw_joints --mode win46 > std_pnp5_yaw.log 2>&1 &)
sleep 1
