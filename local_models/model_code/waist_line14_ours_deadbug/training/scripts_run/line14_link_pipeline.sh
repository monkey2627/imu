#!/bin/bash
# Line 14 "link" pipeline (C20 estimators + C21 downstream): stronger PivotPrint = sensor-to-bone link (orientation offsets + geometry), estimated by ONE
# calibration encoder, evaluated in a wide-scale (0.5-1.2) population with per-identity sensor orientation offsets (RMS 5 deg) and lever arms (3 cm).
#
# run on the server, from anywhere:
#   setsid nohup bash /home/duanyuhan/dyh/motion/lines/line14/scripts_run/line14_link_pipeline.sh > /home/duanyuhan/dyh/motion/lines/line14/logs/line14_link_pipeline.log 2>&1 < /dev/null &
# stages (default: all, resumable - finished outputs are skipped):  STAGES="data smoke est assemble e2e summary"  bash line14_link_pipeline.sh
# optional env:  GPUS="0,1,2,3,4,6,7"  WORKERS=24  SEEDS="42 7 3"
set -u
set -o pipefail
cd /home/duanyuhan/dyh/motion/lines/line14 || exit 1
export PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 LINE9_GAUGE=1 MOBILEPOSER_FPS=60
PY=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
STAGES=${STAGES:-"data smoke est assemble e2e summary"}
GPUS=${GPUS:-"0,1,2,3,4,6,7"}
WORKERS=${WORKERS:-24}
SEEDS=${SEEDS:-"42 7 3"}
VIRT=data_local/line15_cal_virtual
IDS=data_local/line15_cal_ids
L=logs/line14/C20; mkdir -p $L logs/line14/C21
has() { [[ " $STAGES " == *" $1 "* ]]; }
say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { say "FAILED: $*"; exit 1; }

say "files / disk"
for f in line14_link_estimator.py line14_assemble_link.py line14_e2e_link.py line14_poser_chain.py line14_sched.py line14_build_cal_set.py line14_build_cal_identities.py line14_C21_summarize.py; do
  [ -f scripts_run/$f ] || die "missing scripts_run/$f (copy the bundle first)"; done
for k in 0 1 2 3 4; do [ -f data_local/folds/fold$k.json ] || die "missing data_local/folds/fold$k.json"; done
df -h /home/duanyuhan | tail -1; du -sh data_local | tail -1

# ---------------------------------------------------------------- 1. calibration data with persistent sensor offsets
if has data; then
  export LINE14_PROTO=dyn LINE14_STRIDE=2
  if [ ! -f $VIRT/val.pt ]; then
    say "build virtual subjects (40000, offsets RMS 5 deg, lever 3 cm, dynamic 13 s protocol)"
    $PY scripts_run/line14_build_cal_set.py --out $VIRT --n-train 40000 --val-sessions 8 --rot-deg 5.0 --lever-cm 3.0 --dev-deg 10.0 --workers $WORKERS > $L/build_virtual.log 2>&1 || die "build virtual (see $L/build_virtual.log)"
  fi
  if [ ! -f $IDS/val.pt ]; then
    say "build identity sessions (offset_key = identity = the offsets of data/line9_store_calib6)"
    $PY scripts_run/line14_build_cal_identities.py --out $IDS --train-sessions 8 --val-sessions 4 --rot-deg 5.0 --lever-cm 3.0 --dev-deg 10.0 --workers $WORKERS > $L/build_ids.log 2>&1 || die "build ids (see $L/build_ids.log)"
  fi
  unset LINE14_PROTO LINE14_STRIDE
  say "data ready"; du -sh $VIRT $IDS
fi

# ---------------------------------------------------------------- 2. smoke tests (tiny subsets; nothing is kept)
if has smoke; then
  G0=$(echo $GPUS | cut -d, -f1)
  SM="--data $VIRT --seed 42 --gpu $G0 --tag C20_smoke --smoke --rot-deg 5 --scale-range 0.5,1.2 --ids-scale 0.5,1.2 --predict $IDS/val.pt --pred-name smoke"
  for m in rot geo uni; do
    say "smoke $m"; $PY scripts_run/line14_link_estimator.py --mode $m $SM > $L/smoke_$m.log 2>&1 || { tail -15 $L/smoke_$m.log; die "smoke $m"; }
    grep -v Warn $L/smoke_$m.log | tail -2
  done
  say "smoke cas"; $PY scripts_run/line14_link_estimator.py --mode cas --stage1 checkpoints/line14/C20_smoke/rot_s42_smoke.pt $SM > $L/smoke_cas.log 2>&1 || { tail -15 $L/smoke_cas.log; die "smoke cas"; }
  grep -v Warn $L/smoke_cas.log | tail -2
  say "smoke downstream"; $PY scripts_run/line14_e2e_link.py --ori est --ori-file /home/duanyuhan/dyh/motion/SparseIMU/results/line9/enc_v2/pred_phys_h128_big_s42.pt --geom height       --seed 42 --gpu $G0 --tag C21_smoke --smoke --scale-range 0.5,1.2 > $L/smoke_e2e.log 2>&1 || { tail -15 $L/smoke_e2e.log; die "smoke e2e"; }
  grep -v Warn $L/smoke_e2e.log | tail -2
  rm -f $IDS/pred_*_smoke_*.pt; rm -rf results/line14/C20_smoke results/line14/C21_smoke checkpoints/line14/C20_smoke checkpoints/line14/C21_smoke
  say "smoke OK"
fi

# ---------------------------------------------------------------- 3. estimators: 4 modes x (5 cross-fit folds + 1 full model)
if has est; then
  EST="--data $VIRT --seed 42 --tag C20 --rot-deg 5 --scale-range 0.5,1.2 --ids-scale 0.5,1.2 --epochs 40"
  : > $L/jobs_A.txt; : > $L/jobs_B.txt
  for m in rot geo uni; do
    for k in 0 1 2 3 4; do
      [ -f checkpoints/line14/C20/${m}_s42_fold$k.pt ] && [ -f $IDS/pred_${m}_s42_fold${k}_train.pt ] || \
        echo "$PY scripts_run/line14_link_estimator.py --mode $m $EST --gpu {GPU} --exclude-ids data_local/folds/fold$k.json --pred-name fold$k --predict $IDS/train.pt" >> $L/jobs_A.txt
    done
    [ -f $IDS/pred_${m}_s42_full_val.pt ] || echo "$PY scripts_run/line14_link_estimator.py --mode $m $EST --gpu {GPU} --pred-name full --predict $IDS/val.pt" >> $L/jobs_A.txt
  done
  say "estimators A (rot, geo, uni): $(wc -l < $L/jobs_A.txt) jobs"
  [ -s $L/jobs_A.txt ] && { $PY scripts_run/line14_sched.py $L/jobs_A.txt --gpus $GPUS --log-dir $L/logs_A > $L/sched_A.log 2>&1 || die "estimators A (see $L/sched_A.log)"; }
  for k in 0 1 2 3 4; do
    [ -f $IDS/pred_cas_s42_fold${k}_train.pt ] || \
      echo "$PY scripts_run/line14_link_estimator.py --mode cas $EST --gpu {GPU} --stage1 checkpoints/line14/C20/rot_s42_fold$k.pt --exclude-ids data_local/folds/fold$k.json --pred-name fold$k --predict $IDS/train.pt" >> $L/jobs_B.txt
  done
  [ -f $IDS/pred_cas_s42_full_val.pt ] || echo "$PY scripts_run/line14_link_estimator.py --mode cas $EST --gpu {GPU} --stage1 checkpoints/line14/C20/rot_s42_full.pt --pred-name full --predict $IDS/val.pt" >> $L/jobs_B.txt
  say "estimators B (cascade): $(wc -l < $L/jobs_B.txt) jobs"
  [ -s $L/jobs_B.txt ] && { $PY scripts_run/line14_sched.py $L/jobs_B.txt --gpus $GPUS --log-dir $L/logs_B > $L/sched_B.log 2>&1 || die "estimators B (see $L/sched_B.log)"; }
  say "estimators done"
fi

# ---------------------------------------------------------------- 4. assemble cross-fitted estimates per identity + report the estimation errors
if has assemble; then
  for m in rot geo uni cas; do
    say "assemble $m"; $PY scripts_run/line14_assemble_link.py $IDS $m 42 data_local/line15_est_$m.pt data_local/line15_R_$m.pt > $L/assemble_$m.txt 2>&1 || { tail -15 $L/assemble_$m.txt; die "assemble $m"; }
    grep -v Warn $L/assemble_$m.txt
  done
fi

# ---------------------------------------------------------------- 5. downstream end-to-end (wide scale + offsets), 8 arms x seeds
if has e2e; then
  W="--tag C21 --scale-range 0.5,1.2 --epochs 40"
  : > $L/jobs_C.txt
  add() { # ori geom suffix extra...
    ori=$1; geom=$2; suf=$3; shift 3
    for s in $SEEDS; do
      f=results/line14/C21/ori-${ori}_geom-${geom}_s${s}${suf:+_$suf}.json
      [ -f $f ] || echo "$PY scripts_run/line14_e2e_link.py --ori $ori --geom $geom --seed $s --gpu {GPU} $W ${suf:+--name-suffix $suf} $*" >> $L/jobs_C.txt
    done
  }
  add none none ""
  add none height ""
  add est none rot --ori-file data_local/line15_R_rot.pt
  add est estp6 sep --ori-file data_local/line15_R_rot.pt --est-file data_local/line15_est_geo.pt
  add est estp6 uni --ori-file data_local/line15_R_uni.pt --est-file data_local/line15_est_uni.pt
  add est estp6 cas --ori-file data_local/line15_R_rot.pt --est-file data_local/line15_est_cas.pt
  add true none ""
  add true height ""
  say "downstream: $(wc -l < $L/jobs_C.txt) jobs"
  [ -s $L/jobs_C.txt ] && { $PY scripts_run/line14_sched.py $L/jobs_C.txt --gpus $GPUS --log-dir logs/line14/C21 > $L/sched_C.log 2>&1 || die "downstream (see $L/sched_C.log)"; }
fi

# ---------------------------------------------------------------- 6. summary
if has summary; then
  say "summary"; $PY scripts_run/line14_C21_summarize.py 2>&1 | tee $L/C21_summary.txt
fi
say "ALL DONE"
