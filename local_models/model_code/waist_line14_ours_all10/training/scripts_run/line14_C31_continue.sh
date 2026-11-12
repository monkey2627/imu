#!/usr/bin/env bash
set -u
ROOT=/home/duanyuhan/dyh/motion/lines/line14
PID=${1:-3005364}
LOG="$ROOT/logs/line14/C31_continue.log"
source /home/duanyuhan/SoftWare/miniconda3/etc/profile.d/conda.sh
conda activate mobileposer
cd "$ROOT"
mkdir -p "$ROOT/logs/line14" "$ROOT/results/line14"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for C31 pid=$PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] C31 exited"
BASE="$ROOT/results/line14/C31_real.json"
if [ ! -s "$BASE" ]; then echo "C31 result missing"; exit 2; fi
python - "$BASE" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); rows=p.get('rows',{})
assert len(rows.get('dipimu',[])) == 19, len(rows.get('dipimu',[]))
assert len(rows.get('totalcapture_officalib',[])) == 34, len(rows.get('totalcapture_officalib',[]))
print('C31 completeness OK')
PY
for ns in 4 1; do
  OUT="$ROOT/results/line14/C31_real_ns${ns}.json"
  echo "[$(date -Is)] sensitivity ns=$ns"
  CUDA_VISIBLE_DEVICES= PYTHONPATH="$ROOT" MOBILEPOSER_FPS=60 \
    python "$ROOT/scripts_run/line14_C31_realeval.py" --mc 8 --ns "$ns" --out "$OUT"
done
echo "[$(date -Is)] sensitivity runs complete; manual summary gate remains"
