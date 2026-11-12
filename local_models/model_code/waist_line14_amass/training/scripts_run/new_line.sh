#!/bin/bash
# Create an isolated research-line workspace from the frozen baseline.
# usage: bash /home/duanyuhan/dyh/motion/lines/baseline_20261010/scripts_run/new_line.sh <line_name>
# Result: /home/duanyuhan/dyh/motion/lines/<line_name>/
#   code / scripts_run / docs / third_party : independent copy (edit freely, never touches other lines)
#   checkpoints/ : own real dir; baseline checkpoints hard-linked in (no extra space); new ones stay local
#   data/        : symlink to the shared READ-ONLY data (do not write into it)
#   data_local/  : put line-specific caches here (MOBILEPOSER_DATA_CACHE=data_local/xxx.pt)
#   results/ logs/ : own, empty
set -e
NAME=${1:?line name required}
L=/home/duanyuhan/dyh/motion/lines
B=$L/baseline_20261010
D=$L/$NAME
[ -e "$D" ] && { echo "$D already exists, refusing to overwrite"; exit 1; }
mkdir -p "$D"
cd "$B"
cp -a mobileposer scripts_run docs third_party drift_eval_common.py README.md requirements.txt setup.py LICENSE "$D"/
find "$D" -name __pycache__ -prune -exec rm -rf {} +
cp -al "$(readlink -f "$B/checkpoints")" "$D/checkpoints"
ln -s "$(readlink -f "$B/data")" "$D/data"
mkdir -p "$D/data_local" "$D/results/standard" "$D/logs"
echo "created $D"
