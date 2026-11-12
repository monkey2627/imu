cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
for ds in EKUT TotalCapture; do
  OMP_NUM_THREADS=4 MOBILEPOSER_FPS=60 MOBILEPOSER_RESAMPLE_ANY_FPS=1 nohup $P -m mobileposer.no_head_layouts --layout pnp5 --imu-mode surface --output-root data/pnp5_60_processed --datasets $ds > logs_gen60/$ds.log 2>&1 &
done
sleep 45
tail -c 200 logs_gen60/TotalCapture.log | tr "\r" "\n" | tail -2
