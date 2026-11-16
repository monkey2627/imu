"""Held-out dead-bug leg metrics under acceleration ablations, through the realtime StreamAdapter.

Conditions: synthetic input as trained; acceleration set to zero; acceleration negated.
Everything else (orientations, checkpoint, schedule) is unchanged.  Run with the imudino env.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

E = Path("/home/duanyuhan/dyh/motion/lines/imudino_deadbug_eval_20261114")
N = Path("/home/duanyuhan/dyh/motion/lines/imudino_deadbug_20261113")
sys.path.insert(0, str(E / "model_code"))
from waist_imudino_fiveimu.stream_entry import StreamAdapter  # noqa: E402

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
WARMUP = 60


def globals_of(local):
    g = [None] * 24
    for j, p in enumerate(PARENTS):
        g[j] = local[:, j] if p < 0 else g[p] @ local[:, j]
    return np.stack(g, 1)


def legs(local):
    g = globals_of(local)
    spine = g[:, 3]
    a_axis, b_axis = -spine[:, :, 1], spine[:, :, 2]
    out = {}
    for side, hip, knee in (("L", 1, 4), ("R", 2, 5)):
        th, sh = -g[:, hip, :, 1], -g[:, knee, :, 1]
        out[side + "_thigh"] = np.degrees(np.arctan2((th * b_axis).sum(-1), (th * a_axis).sum(-1)))
        out[side + "_knee"] = np.degrees(np.arccos(np.clip((th * sh).sum(-1), -1, 1)))
    return out


def main():
    cache = torch.load(N / "data/features/stream_wristankle_v1/DeadBug.stream.pt", map_location="cpu",
                       weights_only=False)["sequences"]
    data = torch.load(N / "data/processed/wrists_shanks_waist/DeadBug.pt", map_location="cpu", weights_only=False)
    preds = {p["cache_index"]: p for p in torch.load(E / "out/heldout_predictions.pt", weights_only=False)}
    adapter = StreamAdapter(E / "ckpt/best.pt", device="cuda:0")   # best.pt + imudino.pt symlinks
    used = set()
    results = {}
    for name, transform in (("synthetic", lambda a: a), ("accel_zero", lambda a: a * 0.0),
                            ("accel_negated", lambda a: -a)):
        errs = {k: [] for k in ("L_thigh", "R_thigh", "L_knee", "R_knee")}
        ext_hit = {"L": [0, 0], "R": [0, 0]}
        for ci, pred in preds.items():
            seq = cache[ci]
            match = next(i for i, (o, g) in enumerate(zip(data["ori"], data["groups"]))
                         if i not in used and g == seq["group"] and len(o) == len(seq["ori"])
                         and torch.allclose(o[:, :5].float(), seq["ori"].float(), atol=1e-5))
            ori = data["ori"][match][:, :5].float().numpy()
            acc = transform(data["acc"][match][:, :5].float().numpy())
            adapter.reset()
            outs = {}
            for f in range(len(ori)):
                adapter.push(None, ori[f], acc[f])
                if f >= 1:
                    outs[adapter.pushed - 2] = adapter.infer().numpy()
            idx = [k for k in sorted(outs) if k >= WARMUP]
            p_l, t_l = legs(np.stack([outs[k] for k in idx])), legs(pred["gt_local"][idx])
            for k in errs:
                errs[k].append(np.abs(p_l[k] - t_l[k]).mean())
            for side in "LR":
                gt_ext = (t_l[side + "_thigh"] < 45) & (t_l[side + "_knee"] < 40)
                pr_ext = (p_l[side + "_thigh"] < 45) & (p_l[side + "_knee"] < 40)
                ext_hit[side][0] += int((gt_ext & pr_ext).sum())
                ext_hit[side][1] += int(gt_ext.sum())
        results[name] = {k + "_abs_err_deg": round(float(np.mean(v)), 2) for k, v in errs.items()}
        results[name].update({f"{s}_extended_recall": round(ext_hit[s][0] / max(ext_hit[s][1], 1), 3) for s in "LR"})
        print(name, json.dumps(results[name]), flush=True)
    (E / "out/accel_ablation.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
