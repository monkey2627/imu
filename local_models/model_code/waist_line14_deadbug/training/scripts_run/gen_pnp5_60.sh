cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
mkdir -p logs_gen60
for ds in ACCAD BioMotionLab_NTroje BMLhandball BMLmovi CMU DanceDB DFaust_67 Eyes_Japan_Dataset HUMAN4D HumanEva KIT MPI_HDM05 MPI_Limits MPI_mosh SFU SSM_synced TCD_handMocap Transitions_mocap; do
  OMP_NUM_THREADS=4 MOBILEPOSER_FPS=60 MOBILEPOSER_RESAMPLE_ANY_FPS=1 nohup $P -m mobileposer.no_head_layouts --layout pnp5 --imu-mode surface --output-root data/pnp5_60_processed --datasets $ds > logs_gen60/$ds.log 2>&1 &
done
sleep 60; tail -n 2 logs_gen60/HumanEva.log logs_gen60/ACCAD.log | cut -c1-200; echo launched
