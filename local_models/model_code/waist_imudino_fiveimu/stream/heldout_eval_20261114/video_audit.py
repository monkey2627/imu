"""(1) Held-out leg metrics without 34.mp4; (2) per-source-video leg pattern audit of the DeadBug set.

Standard dead bug alternates legs: one leg extends while the other stays in tabletop, so frames with
BOTH legs extended should be rare and each leg should extend on its own.
"""
import json
from pathlib import Path

import numpy as np
import torch

N = Path("/home/duanyuhan/dyh/motion/lines/imudino_deadbug_20261113")
E = Path("/home/duanyuhan/dyh/motion/lines/imudino_deadbug_eval_20261114")
RAW = Path("/home/duanyuhan/dyh/motion/data_system/runtime/output/training_data/DeadBug")
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def sixd(v):
    a, b = v[..., :3], v[..., 3:6]
    c1 = torch.nn.functional.normalize(a, dim=-1)
    b = b - (c1 * b).sum(-1, keepdim=True) * c1
    c2 = torch.nn.functional.normalize(b, dim=-1)
    return torch.stack((c1, c2, torch.cross(c1, c2, dim=-1)), -1)


def leg_angles(local23):
    """local23: (T,23,3,3) joints 1..23; spine-relative thigh angle and knee flexion per side."""
    T = len(local23)
    local = torch.cat((torch.eye(3).expand(T, 1, 3, 3), local23), 1)
    g = [None] * 24
    for j, p in enumerate(PARENTS):
        g[j] = local[:, j] if p < 0 else g[p] @ local[:, j]
    spine = g[3]
    a_axis, b_axis = -spine[..., :, 1], spine[..., :, 2]
    out = {}
    for s, hip, knee in (("L", 1, 4), ("R", 2, 5)):
        th, sh = -g[hip][..., :, 1], -g[knee][..., :, 1]
        out[s + "_thigh"] = torch.rad2deg(torch.atan2((th * b_axis).sum(-1), (th * a_axis).sum(-1))).numpy()
        out[s + "_knee"] = torch.rad2deg(torch.arccos((th * sh).sum(-1).clamp(-1, 1))).numpy()
    return out


def main():
    names = {}
    for job in RAW.iterdir():
        for meta in job.glob("clip*.json"):
            m = json.loads(meta.read_text())
            names[m["source_video_sha256"]] = (m["source_video"]["filename"], job.name)
    seqs = torch.load(N / "data/features/stream_wristankle_v1/DeadBug.stream.pt", map_location="cpu",
                      weights_only=False)["sequences"]
    per_video = {}
    for s in seqs:
        a = leg_angles(sixd(s["target"].float().reshape(-1, 23, 6)))
        ext = {k: (a[k + "_thigh"] < 45) & (a[k + "_knee"] < 40) for k in "LR"}
        flex = {k: (a[k + "_thigh"] > 60) & (a[k + "_knee"] > 50) for k in "LR"}
        v = per_video.setdefault(s["group"].split(":")[1], {"clips": 0, "frames": 0, "L_ext": 0, "R_ext": 0, "both_ext": 0,
                                                             "one_ext_other_flex": 0, "both_flex": 0, "val": bool(s["val"])})
        v["clips"] += 1
        v["frames"] += len(a["L_thigh"])
        v["L_ext"] += int(ext["L"].sum())
        v["R_ext"] += int(ext["R"].sum())
        v["both_ext"] += int((ext["L"] & ext["R"]).sum())
        v["one_ext_other_flex"] += int(((ext["L"] & flex["R"]) | (ext["R"] & flex["L"])).sum())
        v["both_flex"] += int((flex["L"] & flex["R"]).sum())
    rows = []
    for sha, v in per_video.items():
        any_ext = v["L_ext"] + v["R_ext"] - v["both_ext"]
        rows.append({"video": names.get(sha, ("?", ""))[0], "sha": sha[:12], "split": "heldout" if v["val"] else "train",
                     "clips": v["clips"], "seconds": round(v["frames"] / 60, 1),
                     "ext_share": round(any_ext / v["frames"], 3),
                     "both_ext_share_of_ext": round(v["both_ext"] / max(any_ext, 1), 3),
                     "alternating_share_of_ext": round(v["one_ext_other_flex"] / max(any_ext, 1), 3),
                     "both_flex_share": round(v["both_flex"] / v["frames"], 3)})
    rows.sort(key=lambda r: -r["both_ext_share_of_ext"])
    print(f"{'video':10s} {'split':8s} clips  sec   ext%  bothExt/ext  alternating/ext  tabletop%")
    for r in rows:
        print(f"{r['video']:10s} {r['split']:8s} {r['clips']:4d} {r['seconds']:6.1f} {100*r['ext_share']:5.1f}"
              f"   {100*r['both_ext_share_of_ext']:6.1f}       {100*r['alternating_share_of_ext']:6.1f}       {100*r['both_flex_share']:5.1f}")
    (E / "out/video_leg_audit.json").write_text(json.dumps(rows, indent=1))

    # Held-out metrics without 34.mp4, from the per-sequence report + ablation-style recall
    report = json.loads((E / "out/deadbug_holdout_report.json").read_text())
    sha34 = next(s for s, (n, _) in names.items() if n == "34.mp4")
    keep = [r for r in report["heldout_per_sequence"] if not r["group"].endswith(sha34)]
    w = np.array([r["frames"] for r in keep], float)
    print("\nheld-out without 34.mp4:", len(keep), "sequences")
    for k in ("local", "L_thigh_abs_err", "R_thigh_abs_err", "L_knee_abs_err", "R_knee_abs_err"):
        print(f"  {k}: mean over sequences {np.mean([r[k] for r in keep]):.2f}  (frame-weighted {np.average([r[k] for r in keep], weights=w):.2f})")
    idx = json.loads((E / "overlay/index.json").read_text())
    hit = {"L": [0, 0], "R": [0, 0]}
    for it in idx:
        if it["video"] == "34.mp4":
            continue
        d = np.load(E / "overlay" / it["file"])
        for s in "LR":
            g = (d[f"gt_{s}_thigh"] < 45) & (d[f"gt_{s}_knee"] < 40)
            p = (d[f"pred_{s}_thigh"] < 45) & (d[f"pred_{s}_knee"] < 40)
            g[:30] = False
            hit[s][0] += int((g & p).sum())
            hit[s][1] += int(g.sum())
    print("  extended-frame recall L/R:", round(hit["L"][0] / hit["L"][1], 3), round(hit["R"][0] / hit["R"][1], 3),
          " (gt extended frames", hit["L"][1], hit["R"][1], ")")


if __name__ == "__main__":
    main()
