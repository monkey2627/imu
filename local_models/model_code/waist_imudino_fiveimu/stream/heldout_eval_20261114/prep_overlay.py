"""Map held-out cache sequences to their DuoMo clips and write GT / predicted SMPL vertices (30 fps).

Run (mobileposer env, same PYTHONPATH as eval_deadbug_holdout.py).
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths

E = Path("/home/duanyuhan/dyh/motion/lines/imudino_deadbug_eval_20261114")
RAW = Path("/home/duanyuhan/dyh/motion/data_system/runtime/output/training_data/DeadBug")
CACHE = Path("/home/duanyuhan/dyh/motion/lines/imudino_deadbug_20261113/data/features/stream_wristankle_v1/DeadBug.stream.pt")


def sixd(values):
    a, b = values[..., :3], values[..., 3:6]
    c1 = torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
    b = b - (c1 * b).sum(-1, keepdim=True) * c1
    c2 = torch.nn.functional.normalize(b, dim=-1, eps=1e-8)
    return torch.stack((c1, c2, torch.cross(c1, c2, dim=-1)), dim=-1)


def leg_text_angles(local):
    g = [None] * 24
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    for j, p in enumerate(parents):
        g[j] = local[:, j] if p < 0 else g[p] @ local[:, j]
    spine = g[3]
    a_axis, b_axis = -spine[:, :, 1], spine[:, :, 2]
    out = {}
    for side, hip, knee in (("L", 1, 4), ("R", 2, 5)):
        th, sh = -g[hip][:, :, 1], -g[knee][:, :, 1]
        out[side + "_thigh"] = np.degrees(np.arctan2((th * b_axis).sum(-1), (th * a_axis).sum(-1)))
        out[side + "_knee"] = np.degrees(np.arccos(np.clip((th * sh).sum(-1), -1, 1)))
    return out


def main():
    preds = torch.load(E / "out/heldout_predictions.pt", map_location="cpu", weights_only=False)
    seqs = torch.load(CACHE, map_location="cpu", weights_only=False)["sequences"]
    clips = []
    for job in sorted(RAW.iterdir()):
        for meta_path in sorted(job.glob("clip*.json")):
            meta = json.loads(meta_path.read_text())
            clips.append((job.name, meta, job / meta_path.name.replace(".json", "_poses.npz")))
    body = ParametricModel(paths.smpl_file, device="cpu")
    index = []
    for item in preds:
        seq = seqs[item["cache_index"]]
        sha = item["group"].split(":", 1)[1]
        target = sixd(seq["target"].float().reshape(-1, 23, 6))[:, :21].numpy()
        match = None
        for job, meta, npz in clips:
            if meta["source_video_sha256"] != sha or meta["num_frames"] != len(target):
                continue
            raw = np.load(npz)
            rot = Rotation.from_rotvec(raw["poses"][:, 3:66].reshape(-1, 3)).as_matrix().reshape(-1, 21, 3, 3)
            err = np.abs(rot - target).max()
            if err < 1e-3:
                match = (job, meta, raw)
                break
        if match is None:
            raise RuntimeError(f"no raw clip for cache sequence {item['cache_index']}")
        job, meta, raw = match
        betas = torch.from_numpy(raw["betas"][:10].astype(np.float32))
        frames30 = np.arange(0, len(target), 2)
        meshes = {}
        for name, key in (("gt", "gt_local"), ("pred", "pred_local")):
            local = torch.from_numpy(item[key][frames30])
            verts = body.forward_kinematics(local, shape=betas.expand(len(local), -1), calc_mesh=True)[2]
            meshes[name] = verts.numpy().astype(np.float32)
        angles = {f"{who}_{k}": v[frames30].astype(np.float32)
                  for who, key in (("gt", "gt_local"), ("pred", "pred_local"))
                  for k, v in leg_text_angles(item[key]).items()}
        out = E / "overlay" / f"seq{item['cache_index']:03d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, gt=meshes["gt"].astype(np.float16), pred=meshes["pred"].astype(np.float16),
                            **angles)
        index.append({"cache_index": item["cache_index"], "job": job, "clip_index": meta["clip_index"],
                      "video": meta["source_video"]["filename"], "frames30": len(frames30), "file": out.name})
        print(index[-1], flush=True)
    (E / "overlay/index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
