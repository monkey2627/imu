"""Mesh contact quality per tag: palm/knee lowest-vertex heights (renderer grounding = lowest vertex on y=0),
horizontal drift of wrists/knees (pelvis-root frame, as Unity shows it), per-frame pelvis height jumps.
    python -m analysis.scratch_lyingcat.contacts <stem> <tag> [<tag> ...]"""
import sys, numpy as np, torch
from analysis.compare_video import smpl_model
import production.mesh_video as mv
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
torch.set_grad_enabled(False)
model, _ = smpl_model()
dom = model["bone_ids"][np.arange(len(model["bone_ids"])), model["weights"].argmax(1)]
REG = {"handL": [20, 22], "handR": [21, 23], "kneeL": [4], "kneeR": [5]}
masks = {k: np.isin(dom, v) for k, v in REG.items()}


def evaluate(stem, tag, t0=9.0, t1=None):
    d = np.load(f"analysis/scratch_lyingcat/cap_{stem}_{tag}.npz")
    t = d["t"]; t1 = t1 or (51.5 if "230158" in stem else 56.5)       # before getting up
    idx = np.where((t >= t0) & (t <= t1))[0]
    H, J = [], []
    for i in idx:
        q = batch_smpl_matrix_to_unity_quat_xyzw(d["final"][i].astype(np.float32))
        v, j = mv.skin(model, q, True, return_joints=True)
        H.append([100 * v[m, 1].min() for m in masks.values()]); J.append(j)
    H, J = np.array(H), np.array(J) * 100
    out = {"t": t[idx], "H": H}
    spread = H.max(1) - H.min(1)                        # 4 supports co-planar -> 0
    horiz = {name: J[:, jt][:, [0, 2]] for name, jt in (("wristL", 20), ("wristR", 21), ("kneeL", 4), ("kneeR", 5))}
    drift = {k: np.linalg.norm(v - np.median(v, 0), axis=1) for k, v in horiz.items()}
    dy = np.abs(np.diff(J[:, 0, 1]))
    print(f"{stem} [{tag}] {t0}-{t1}s  heights cm (p50/p95): " +
          "  ".join(f"{k} {np.median(H[:, c]):.1f}/{np.percentile(H[:, c], 95):.1f}" for c, k in enumerate(REG)) +
          f" | support spread p50 {np.median(spread):.1f} p95 {np.percentile(spread, 95):.1f} max {spread.max():.1f}")
    print("      horiz drift from median cm (p95/max): " + "  ".join(
        f"{k} {np.percentile(v, 95):.1f}/{v.max():.1f}" for k, v in drift.items()) +
          f" | pelvis |dy| per frame p99 {np.percentile(dy, 99):.2f} max {dy.max():.2f} cm, n>1cm {(dy > 1).sum()}")
    return out


if __name__ == "__main__":
    for tag in sys.argv[2:]: evaluate(sys.argv[1], tag)
