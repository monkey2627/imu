"""Lightweight real-data non-degradation check (line 14, stage 5 light).

Real DIP-IMU test split and TotalCapture (DIP / official calibration) from the PNP test sets, 5 IMUs (no head), 60 fps.
Models: the line-14 Joints regressors trained on synthetic AMASS only (no DIP fine-tuning, shape-aware labels):
  B0            : E5 r1_b0   (no geometry input)
  FiLM-pca3 true: E10f pca3_oracle_film, geometry = population mean  (the 'gate closed' / no-calibration case)
  FiLM-estp3    : E14  estp3_oracle_film (trained with cross-fitted calibration estimates), geometry = population mean
Because DIP/TC have no calibration sequences and no known body shape, the geometry input is set to the training-population mean
(normalised vector 0).  This measures only whether the geometry conditioning HURTS real-data accuracy when no calibration is given.
Metric: root-relative joint position error (cm), mean over frames and joints (and over 'unsensed' joints), against FK(GT pose, mean shape);
per-sequence means, averaged over sequences; paired bootstrap over sequences for the difference to B0.
"""
from __future__ import annotations
import os
os.environ["MOBILEPOSER_FPS"] = "60"
import argparse, json
from pathlib import Path
import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.evaluate_standard import sequences, load_pnp_articulate
from mobileposer.line9.models import Line9Joints, GeometryStats
from mobileposer.line9.evaluation import JOINT_GROUPS
from mobileposer.line14.models import FilmJoints

DATASETS = ["dipimu", "totalcapture_dipcalib", "totalcapture_officalib"]
WIN = 125
SPECS = {   # label: (checkpoint pattern, kind)
    "B0": ("checkpoints/line14/E5/r1_b0_s{s}.pt", "b0"),
    "FiLM-pca3(true)": ("checkpoints/line14/E10f/pca3_oracle_film_s{s}.pt", "film"),
    "FiLM-estp3": ("checkpoints/line14/E14/estp3_oracle_film_s{s}.pt", "film"),
}


def load(pattern, kind, seed, dev):
    st = torch.load(pattern.format(s=seed), map_location=dev, weights_only=False)
    if kind == "b0":
        m = Line9Joints(mode="b0", geom_dim=st["geometry_mean"].numel())
    else:
        m = FilmJoints(st["geometry_mean"].numel(), GeometryStats(mean=st["geometry_mean"], std=st["geometry_std"]))
    m.load_state_dict(st["model"]); return m.to(dev).eval()


@torch.no_grad()
def predict(model, kind, imu, dev):
    T = len(imu); chunks = [(s, min(s + WIN, T)) for s in range(0, T, WIN)]
    L = [e - s for s, e in chunks]
    x = torch.zeros(len(chunks), max(L), 60)
    for k, (s, e) in enumerate(chunks): x[k, : e - s] = imu[s:e]
    g = None if kind == "b0" else model.geometry_mean.unsqueeze(0).expand(len(chunks), -1)
    p = model(x.to(dev), L, None if g is None else g.to(dev)).float().cpu()
    return torch.cat([p[k, : L[k]] for k in range(len(chunks))]).reshape(T, 24, 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--out", type=Path, default=Path("results/line14/real_check.json"))
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    art = load_pnp_articulate(); body = ParametricModel(paths.smpl_file)
    models = {lab: {s: load(p, k, s, dev) for s in (42, 7, 3)} for lab, (p, k) in SPECS.items()}
    res = {lab: {ds: {s: [] for s in (42, 7, 3)} for ds in DATASETS} for lab in SPECS}
    names = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        for n, (name, acc, R, pose) in enumerate(sequences(ds, art)):
            if a.smoke and n >= 2: break
            T = len(acc)
            imu = torch.cat([(acc / 30.0).flatten(1), R.flatten(1)], dim=1).float()
            _, gt = body.forward_kinematics(pose=pose.reshape(-1, 216), shape=None, tran=None, calc_mesh=False)
            gt = gt.reshape(T, 24, 3)
            names[ds].append(name)
            for lab, (_, kind) in SPECS.items():
                for s in (42, 7, 3):
                    p = predict(models[lab][s], kind, imu, dev)
                    e = (p - gt).norm(dim=-1) * 100.0                                   # [T,24] cm
                    res[lab][ds][s].append({g: float(e[:, list(idx)].mean()) for g, idx in JOINT_GROUPS.items()})
            print(ds, name, T, flush=True)
    out = {"datasets": {}, "n_seq": {ds: len(names[ds]) for ds in DATASETS}}
    rng = np.random.default_rng(0)
    for ds in DATASETS:
        out["datasets"][ds] = {}
        base = np.array([[r["all"] for r in res["B0"][ds][s]] for s in (42, 7, 3)]).mean(0)           # per-sequence, seed-mean
        for lab in SPECS:
            arr = {g: np.array([[r[g] for r in res[lab][ds][s]] for s in (42, 7, 3)]).mean(0) for g in JOINT_GROUPS}
            d = arr["all"] - base
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            out["datasets"][ds][lab] = {"mean_cm": {g: float(v.mean()) for g, v in arr.items()},
                                        "per_seed_all_cm": [float(np.mean([r["all"] for r in res[lab][ds][s]])) for s in (42, 7, 3)],
                                        "diff_vs_B0_all_cm": float(d.mean()), "diff_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        print(f"\n{ds} ({len(names[ds])} sequences)  root-relative joint position error, cm (3-seed mean):")
        for lab in SPECS:
            r = out["datasets"][ds][lab]
            print(f"  {lab:18s} all {r['mean_cm']['all']:.3f}  unsensed {r['mean_cm']['unsensed']:.3f}  elbow {r['mean_cm']['elbow']:.3f}  knee {r['mean_cm']['knee']:.3f}"
                  f"   diff vs B0 {r['diff_vs_B0_all_cm']:+.3f} [{r['diff_ci95'][0]:+.3f},{r['diff_ci95'][1]:+.3f}]")
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
