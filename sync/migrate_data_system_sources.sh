#!/usr/bin/env bash
set -euo pipefail

MOTION=/home/duanyuhan/dyh/motion
ROOT="$MOTION/data_system"
STAGE="$ROOT/.source-staging"
FLOW="$MOTION/flowhmr"
DUOMO="$MOTION/duomo"
GVHMR="$MOTION/gvhmr"
MP="$MOTION/lines/baseline_20261010"

[[ "$ROOT" == /home/duanyuhan/dyh/motion/data_system ]]
[[ -d "$FLOW/flowhmr" && -d "$DUOMO/src" && -d "$GVHMR/gvhmr" && -d "$MP/mobileposer" ]]
[[ -L "$ROOT/core/flowhmr" && -L "$ROOT/models/duomo" && -L "$ROOT/models/gvhmr" && -L "$ROOT/integrations/mobileposer" ]]

mkdir -p "$STAGE/flowhmr" "$STAGE/duomo" "$STAGE/gvhmr" "$STAGE/mobileposer"

# Copy working trees, including modified and untracked source, but exclude bulky runtime assets.
rsync -a --delete \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='/checkpoints/' --exclude='/ckpts/' --exclude='/output/' --exclude='/assets/' \
  "$FLOW/" "$STAGE/flowhmr/"

rsync -a --delete \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='/data/' --exclude='/.incoming_assets_20261107/' --exclude='/results/' \
  --exclude='/dense_kpt.tar.gz' --exclude='*.tar' --exclude='*.tar.gz' --exclude='*.tar.bz2' \
  "$DUOMO/" "$STAGE/duomo/"

rsync -a --delete \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='/inputs/' --exclude='/outputs/' \
  "$GVHMR/" "$STAGE/gvhmr/"

rsync -a --delete \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='/data' --exclude='/checkpoints' --exclude='/logs/' --exclude='/results/' \
  --exclude='/third_party/' \
  "$MP/" "$STAGE/mobileposer/"
mkdir -p "$STAGE/mobileposer/third_party/PNP"
rsync -a "$MP/third_party/PNP/articulate/" "$STAGE/mobileposer/third_party/PNP/articulate/"
rsync -a "$MP/third_party/PNP/RBDL-PIP/" "$STAGE/mobileposer/third_party/PNP/RBDL-PIP/"

# Include the source package of the comparison dependency, but not its model/data artifacts.
rsync -a --exclude='/data/' --exclude='/models/' --exclude='/drift_results/' \
  "$MP/third_party/PNP/" "$STAGE/mobileposer/third_party/PNP/"

# Require byte-identical copies before changing any legacy source path.
verify_copy() {
  local src="$1" dst="$2"; shift 2
  local diff_file
  diff_file="$(mktemp)"
  rsync -anicO --delete --out-format='%i %n' "$@" "$src/" "$dst/" > "$diff_file"
  if [[ -s "$diff_file" ]]; then
    cat "$diff_file"
    rm -f "$diff_file"
    echo "Source copy verification failed: $src -> $dst" >&2
    exit 2
  fi
  rm -f "$diff_file"
}
verify_copy "$FLOW" "$STAGE/flowhmr" --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='/checkpoints/' --exclude='/ckpts/' --exclude='/output/' --exclude='/assets/'
verify_copy "$DUOMO" "$STAGE/duomo" --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='/data/' --exclude='/.incoming_assets_20261107/' --exclude='/results/' --exclude='/dense_kpt.tar.gz' --exclude='*.tar' --exclude='*.tar.gz' --exclude='*.tar.bz2'
verify_copy "$GVHMR" "$STAGE/gvhmr" --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='/inputs/' --exclude='/outputs/'
verify_copy "$MP" "$STAGE/mobileposer" --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='/data' --exclude='/checkpoints' --exclude='/logs/' --exclude='/results/' --exclude='/third_party/'

for pair in \
  "$ROOT/core/flowhmr:$STAGE/flowhmr" \
  "$ROOT/models/duomo:$STAGE/duomo" \
  "$ROOT/models/gvhmr:$STAGE/gvhmr" \
  "$ROOT/integrations/mobileposer:$STAGE/mobileposer"; do
  old="${pair%%:*}"; staged="${pair#*:}"
  [[ -L "$old" ]]
  rm "$old"
  final="${old%/*}/$(basename "$old")"
  mv "$staged" "$final"
done

# Non-code assets stay at their old locations; the canonical source tree points to them.
ln -s "$FLOW/assets" "$ROOT/core/flowhmr/assets"
ln -s "$FLOW/checkpoints" "$ROOT/core/flowhmr/checkpoints"
ln -s "$FLOW/ckpts" "$ROOT/core/flowhmr/ckpts"
ln -s "$FLOW/output" "$ROOT/core/flowhmr/output"
ln -s "$DUOMO/data" "$ROOT/models/duomo/data"
ln -s "$DUOMO/results" "$ROOT/models/duomo/results"
ln -s "$GVHMR/inputs" "$ROOT/models/gvhmr/inputs"
ln -s "$GVHMR/outputs" "$ROOT/models/gvhmr/outputs"
ln -s "$GVHMR/.venv" "$ROOT/models/gvhmr/.venv"
MP_DATA="$(readlink -f "$MP/data")"
MP_CHECKPOINTS="$(readlink -f "$MP/checkpoints")"
ln -s "$MP_DATA" "$ROOT/integrations/mobileposer/data"
ln -s "$MP_CHECKPOINTS" "$ROOT/integrations/mobileposer/checkpoints"

# Keep old source entrypoints usable as links to the canonical files.
link_old_to_new() {
  local old="$1" new="$2"
  [[ -e "$old" || -L "$old" ]] || return 0
  if [[ -L "$old" && "$(readlink -f "$old")" == "$(readlink -f "$new")" ]]; then return 0; fi
  rm -rf -- "$old"
  ln -s "$new" "$old"
}

for name in flowhmr tools tests configs docs data third_party; do
  link_old_to_new "$FLOW/$name" "$ROOT/core/flowhmr/$name"
done
for name in app.py run_v2m_demo_generation.py train_v2m.py workbench.py workbench.html workbench.js README.md DATA_GENERATION_SYSTEM.md LICENSE requirements.txt requirements-phc.txt .gitignore .gitmodules; do
  link_old_to_new "$FLOW/$name" "$ROOT/core/flowhmr/$name"
done

for name in src scripts configs third_party; do
  link_old_to_new "$DUOMO/$name" "$ROOT/models/duomo/$name"
done
for name in README.md requirements.txt LICENSE.pdf CODE_OF_CONDUCT.md CONTRIBUTING.md inspect_mapping.py inspect_pickle_structure.py inspect_sparse.py; do
  link_old_to_new "$DUOMO/$name" "$ROOT/models/duomo/$name"
done

for name in gvhmr docs scripts space tests tools bin .github; do
  link_old_to_new "$GVHMR/$name" "$ROOT/models/gvhmr/$name"
done
for name in README.md AGENTS.md LICENSE Makefile pyproject.toml pyrightconfig.json uv.lock .gitignore; do
  link_old_to_new "$GVHMR/$name" "$ROOT/models/gvhmr/$name"
done

for name in mobileposer scripts_run docs; do
  link_old_to_new "$MP/$name" "$ROOT/integrations/mobileposer/$name"
done
for name in drift_eval_common.py B0_BASELINE_MANIFEST.json README.md LICENSE requirements.txt setup.py .gitignore; do
  link_old_to_new "$MP/$name" "$ROOT/integrations/mobileposer/$name"
done
# Only the code subtrees are moved; bulky PNP data and model assets remain where they are.
for name in articulate RBDL-PIP; do
  link_old_to_new "$MP/third_party/PNP/$name" "$ROOT/integrations/mobileposer/third_party/PNP/$name"
done

echo SOURCE_MIGRATION_OK
find "$ROOT" -maxdepth 3 -mindepth 1 -printf '%y %p -> %l\n' | sort
