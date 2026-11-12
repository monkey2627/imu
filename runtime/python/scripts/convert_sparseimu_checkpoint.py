#!/usr/bin/env python3
"""Convert a SparseIMU (Lightning) Joints + Poser checkpoint pair into the
single combined ``base_model.pth`` + ``manifest.json`` used by the realtime path.

Run on the training server (the Lightning checkpoints live there)::

    python scripts/convert_sparseimu_checkpoint.py \
        --poser pnp5_60a_s42_ftr3_poser --joints pnp5_60a_s42_ftr3_joints \
        --name pnp5_60a_s42_ftr3
    python scripts/convert_sparseimu_checkpoint.py \
        --poser pnp5_60a_sc_ftr3_poser --joints pnp5_60a_sc_ftr3_joints \
        --name pnp5_60a_sc_ftr3 --anchored

The checkpoint of each module is picked exactly like SparseIMU's
``eval_variant.Pred`` does (``best_ckpt``: lowest ``validation_step_loss``).
Only tensors are written to ``base_model.pth`` (loaded with weights_only=True);
everything else goes into ``manifest.json``.
"""
import argparse
import glob
import hashlib
import json
import re
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPARSEIMU = Path("/home/duanyuhan/dyh/motion/SparseIMU")

# Keep in sync with runtime_contract.py (checked again at load time).
CONTRACT = {"layout": "pnp5_forearm_shank_pelvis", "fps": 60, "window": 91, "outputIndex": 80}
PLACEMENT = {
    "config": "SparseIMU/mobileposer/configs/pnp_surface_imu.json",
    "slots": ["lw", "rw", "ls", "rs", "pelvis"],
    "smplJoints": [18, 19, 4, 5, 0],
    "pnpVertices": [1961, 5424, 1176, 4662, 3021],
    "mountRotation": "identity; orientation = global bone rotation (bone-calibrated)",
}


def best_ckpt(root):
    """Same selection as SparseIMU evaluate_canon_models.best_ckpt (including tie order)."""
    files = glob.glob(f"{root}/*/*/epoch=*.ckpt") + glob.glob(f"{root}/*/epoch=*.ckpt")
    if not files:
        raise FileNotFoundError(f"no epoch=*.ckpt under {root}")

    def loss(f):
        m = re.search(r"validation_step_loss=([0-9.]+)", f)
        return float(m.group(1).rstrip(".")) if m else 1e9

    if all(loss(f) >= 1e9 for f in files):
        return max(files, key=lambda f: int(re.search(r"epoch=([0-9]+)", f).group(1)))
    return min(files, key=loss)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_module(ckpt_path, module, out_dim, in_dim):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = f"{module}."
    out = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            raise ValueError(f"{ckpt_path}: unexpected key {key!r} (expected only {prefix}*)")
        out[f"{module}.{module}.{key[len(prefix):]}"] = value.detach().cpu().contiguous()
    first = out[f"{module}.{module}.linear1.weight"]
    last = out[f"{module}.{module}.linear2.weight"]
    if tuple(first.shape) != (256, in_dim) or tuple(last.shape) != (out_dim, 512):
        raise ValueError(f"{ckpt_path}: unexpected shapes linear1={tuple(first.shape)} linear2={tuple(last.shape)}; "
                         f"expected ({256}, {in_dim}) / ({out_dim}, 512)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparseimu-root", type=Path, default=DEFAULT_SPARSEIMU)
    p.add_argument("--poser", required=True, help="checkpoint dir name under SparseIMU/checkpoints")
    p.add_argument("--joints", required=True)
    p.add_argument("--name", required=True, help="output directory name under --output-root")
    p.add_argument("--anchored", action="store_true", help="the Poser was trained with MOBILEPOSER_SHORTCUT=1 (identity anchoring)")
    p.add_argument("--output-root", type=Path, default=ROOT / "checkpoints")
    args = p.parse_args()

    joints_ckpt = best_ckpt(args.sparseimu_root / "checkpoints" / args.joints)
    poser_ckpt = best_ckpt(args.sparseimu_root / "checkpoints" / args.poser)
    combined = {}
    combined.update(convert_module(joints_ckpt, "joints", 72, 60))
    combined.update(convert_module(poser_ckpt, "pose", 96, 132))

    out_dir = args.output_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "base_model.pth"
    if target.exists():
        raise SystemExit(f"{target} already exists; refusing to overwrite")
    torch.save(combined, target)
    manifest = {
        **CONTRACT,
        "anchored": bool(args.anchored),
        "inputFeature": "acceleration[5x3]/30 then orientation[5x3x3] (60-D), bone-calibrated SMPL frame",
        "placement": PLACEMENT,
        "sources": {
            "joints": {"checkpoint": str(joints_ckpt), "sha256": sha256(joints_ckpt)},
            "poser": {"checkpoint": str(poser_ckpt), "sha256": sha256(poser_ckpt)},
        },
        "sparseimuNames": {"poser": args.poser, "joints": args.joints},
        "baseModelSha256": sha256(target),
        "convertedUnixSeconds": time.time(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    print(f"  joints <- {joints_ckpt}")
    print(f"  poser  <- {poser_ckpt}")


if __name__ == "__main__":
    main()
