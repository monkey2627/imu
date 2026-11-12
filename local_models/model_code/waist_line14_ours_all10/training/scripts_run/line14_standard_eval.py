"""Standard-protocol evaluation for the frozen line-14 pose chains.

Uses the same sequence loader and StdEvaluator as evaluate_standard.py, but
loads the line-14 C21/C23 checkpoints directly. No training or test selection
is performed here; this writes a separate standard result JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import line14_realeval_lib as RL
from line14_physfeat import lever_features
from mobileposer.evaluate_standard import StdEvaluator, sequences
from mobileposer.config import paths
import mobileposer.articulate as art

@torch.no_grad()
def predict_win46(model, imu, r=None, batch_size=256):
    """Predict one pose per frame with PNP's 40-past/5-future protocol."""
    jnet, poser, extra = model
    device = next(jnet.parameters()).device
    T = int(imu.shape[0])
    if T == 0:
        return torch.empty((0, 96), dtype=torch.float32)
    starts = [max(0, t - 40) for t in range(T)]
    ends = [min(T, t + 6) for t in range(T)]
    lengths = [e - s for s, e in zip(starts, ends)]
    max_len = max(lengths)
    chunks = []
    for base in range(0, T, batch_size):
        ss = starts[base:base + batch_size]
        ee = ends[base:base + batch_size]
        ll = lengths[base:base + batch_size]
        x = torch.zeros((len(ss), max_len, 60), dtype=imu.dtype)
        for k, (s, e) in enumerate(zip(ss, ee)):
            x[k, :e - s] = imu[s:e]
        x = x.to(device)
        if extra:
            if r is None:
                raise ValueError("lever-arm feature model requires r")
            rr = r.to(device).expand(len(ss), -1, -1)
            x = torch.cat((x, lever_features(x, rr, ll)), dim=-1)
        joints = jnet(x, ll)
        x_pose = x[:, :joints.shape[1]]
        pose = poser(torch.cat((joints.to(x.dtype), x_pose), dim=-1), ll)
        centers = torch.arange(base, base + len(ss), device=device)
        offsets = torch.tensor(ss, device=device)
        rows = torch.arange(len(ss), device=device)
        chunks.append(pose[rows, centers - offsets].float().cpu())
    return torch.cat(chunks, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("results/standard/line14.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    ev = StdEvaluator(dev)
    body = art.model.ParametricModel(paths.smpl_file, device=dev)
    seeds = (42, 7, 3)
    models = {k: {s: RL.load_model(k, s, dev) for s in seeds} for k in ("b0", "i2s", "lvmean")}
    for model in models["lvmean"].values():
        model[0].input_dim = 96
    mean_r = RL.mean_r()
    datasets = ("dipimu", "totalcapture_dipcalib", "totalcapture_officalib")
    result = {"protocol": {"datasets": datasets, "sequences": "all", "seeds": seeds,
                            "test_accessed": True, "mode": "win46", "past_frames": 40, "future_frames": 5,
                            "sensors": [0, 1, 2, 3, 5], "metrics": "StdEvaluator"},
              "models": {}}
    with torch.no_grad():
        for arm in models:
            result["models"][arm] = {}
            for seed in seeds:
                result["models"][arm][str(seed)] = {}
                for ds in datasets:
                    rows = []
                    for name, acc, ori, pose in sequences(ds, ev.art):
                        if args.smoke and rows:
                            break
                        x = torch.cat([(acc / 30).flatten(1), ori.flatten(1)], 1).float()
                        pred = RL.to_local(body, predict_win46(models[arm][seed], x,
                                                              mean_r if arm == "lvmean" else None), dev)
                        rows.append({"name": str(name), "metrics": ev(pred, pose.to(dev)).tolist()})
                    result["models"][arm][str(seed)][ds] = {
                        "n_sequences": len(rows),
                        "mean": np.mean([r["metrics"] for r in rows], axis=0).tolist(),
                        "rows": rows,
                    }
                    print(arm, seed, ds, result["models"][arm][str(seed)][ds]["n_sequences"],
                          np.round(result["models"][arm][str(seed)][ds]["mean"], 3), flush=True)
    result["mean_over_seeds"] = {}
    for arm in models:
        result["mean_over_seeds"][arm] = {}
        for ds in datasets:
            vals = [result["models"][arm][str(s)][ds]["mean"] for s in seeds]
            result["mean_over_seeds"][arm][ds] = np.mean(vals, axis=0).tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
