#!/bin/bash
# smoke tests for the degradation / mask / scaling code paths (E19-E22)
cd /home/duanyuhan/dyh/motion/lines/line14
export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export LINE14_PROTO=dyn LINE14_STRIDE=2
rm -rf data_local/smoke_degr data_local/smoke_ids
$PY scripts_run/line14_build_cal_set.py --out data_local/smoke_degr --n-train 24 --val-sessions 4 --max-identities 3 --degr-mix --workers 4 2>&1 | tail -1
$PY scripts_run/line14_build_cal_identities.py --out data_local/smoke_ids --degr mix --train-sessions 3 --val-sessions 3 --max-identities 3 --workers 4 2>&1 | tail -1
$PY - <<'EOF'
import torch
t = torch.load("data_local/smoke_degr/train.pt", weights_only=False)
print("keys", sorted(t.keys())); print("amp", t["amp"][:8].tolist()); print("keep sum", t["keep"].sum(1)[:8].tolist())
i = torch.load("data_local/smoke_ids/train.pt", weights_only=False); print("ids amp", i["amp"][:6].tolist(), "keep", i["keep"].sum(1)[:6].tolist())
EOF
$PY scripts_run/line14_estimate_r1.py --smoke --target r2 --stride 2 --gpu 4 --data data_local/smoke_degr --tag smoke --predict data_local/smoke_ids/train.pt --pred-name smoke --ids-scale 0.5,1.2 2>&1 | grep -v Warning | tail -3
ls data_local/smoke_ids
$PY scripts_run/line14_train.py --mode oracle --geom pca3 --seed 42 --gpu 4 --tag smoke --smoke --scale-range 0.5,1.2 --store data_local/line14_store_mount3 2>&1 | grep -v "Warning\|return F.linear\|run_backward" | tail -2
$PY scripts_run/line14_train.py --mode b0 --geom r1 --seed 42 --gpu 4 --tag smoke --smoke --scale-range 0.5,1.2 --store data_local/line14_store_mount3 2>&1 | grep -v "Warning\|return F.linear\|run_backward" | tail -1
rm -rf data_local/smoke_degr data_local/smoke_ids results/line14/smoke checkpoints/line14/smoke
echo SMOKE-DONE
