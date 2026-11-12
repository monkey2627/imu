#!/usr/bin/env python3
"""Clean synthetic 5-IMU data from the coach's Ours motions, in the stream Store format (2026-11-06).

The HumanPose labels (SMPL pose + translation, 60 Hz) of all ten Ours captures are turned into IMU input
exactly like the AMASS synthesis of the waist-surface baselines: orientation = global rotation of the
sensed bone (SMPL 18/19/4/5/3), acceleration = smoothed second difference of the barycentric skin point of
no_head_surface_imu.json (world frame, gravity-free).  Targets as mobileposer.stream.features.targets.
These are the motions AMASS lacks (quadruped leg lifts, supine leg extensions...).  Read-only use of the
existing code; output: data/posacc_ours_synth.pt.

  python build_posacc_ours_synth.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pnp_workspace"))
from mobileposer.articulate.model import ParametricModel      # noqa: E402
from mobileposer.config import paths                            # noqa: E402
from mobileposer.stream.features import targets                 # noqa: E402

DATA = Path("/home/duanyuhan/dyh/motion/dataset/processed/OursHumanPose_20261030")
CFG = json.loads((ROOT / "pnp_workspace/mobileposer/configs/no_head_surface_imu.json").read_text())
LABELS = ["lw", "rw", "ls", "rs", "waist"]


def synth_acc(p, n=4, fps=60):
    acc = torch.zeros_like(p)
    acc[1:-1] = (p[:-2] + p[2:] - 2 * p[1:-1]) * fps ** 2
    if len(p) > 2 * n:
        acc[n:-n] = (p[:-2 * n] + p[2 * n:] - 2 * p[n:-n]) * fps ** 2 / n ** 2
    return acc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="", help="comma-separated Ours actions to leave out (held-out tests)")
    ap.add_argument("--out", default="data/posacc_ours_synth.pt")
    args = ap.parse_args()
    exclude = set(args.exclude.split(",")) - {""}
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    body = ParametricModel(paths.smpl_file, device=dev)
    out = {k: [] for k in ("acc", "ori", "jrel", "r6d")}
    lens, names = [], []
    for cap in sorted((DATA / "captures").iterdir(), key=lambda p: int(p.name)):
        if cap.name in exclude:
            continue
        tgt = torch.load(cap / "targets_humanpose_calibrated_60.pt", map_location="cpu", weights_only=False)
        pose, tran = tgt["pose"].float().to(dev), tgt["tran"].float().to(dev)
        glb, _, verts = body.forward_kinematics(pose, tran=tran, calc_mesh=True)
        pts = []
        for label in LABELS:
            a = CFG["attachments"][label]
            w = torch.tensor(a["weights"], device=dev)
            pts.append((verts[:, a["vertices"]] * w[None, :, None]).sum(1))
        acc = synth_acc(torch.stack(pts, 1))
        ori = glb[:, [CFG["attachments"][l]["joint"] for l in LABELS]]
        jrel, r6d = targets(body, pose)
        out["acc"].append(acc.reshape(len(acc), 15).cpu()); out["ori"].append(ori.reshape(len(ori), 45).cpu())
        out["jrel"].append(jrel.cpu()); out["r6d"].append(r6d.cpu())
        lens.append(len(acc)); names.append(f"ours_synth/{cap.name}")
        print(cap.name, len(acc), "frames", flush=True)
    store = {k: torch.cat(v) for k, v in out.items()}
    store["lens"], store["names"] = torch.tensor(lens), names
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, dest)
    print(dest, int(store["lens"].sum()), "frames")


if __name__ == "__main__":
    main()
