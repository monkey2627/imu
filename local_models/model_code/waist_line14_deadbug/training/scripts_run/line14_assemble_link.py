"""Assemble cross-fitted link estimates (orientation offsets + geometry) for the downstream identities.
train identities: from the fold model that did NOT see them (fold = index % 5 of the sorted train identities); validation identities: from the full model.
usage: python scripts_run/line14_assemble_link.py DIR MODE SEED OUT_EST.pt OUT_R.pt
  DIR      directory with train.pt / val.pt of the identity sessions and the pred_<MODE>_s<SEED>_{fold0..4,full}_{train,val}.pt files
  OUT_EST  {identity: {"mu" [S,24] m, "sigma" [S,24] m, "truth" [24] m, "split"}}   (format of the existing line14_est_r2_*.pt; empty if the model has no geometry head)
  OUT_R    {"R": {identity: [5,3,3]}, "R_true": {...}}   orientation-offset estimate (mean over the identity's sessions, projected to SO(3))"""
import sys
from pathlib import Path
import numpy as np, torch

D, MODE, SEED, OUT_EST, OUT_R = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
ids_train = sorted({i for i in torch.load(D / "train.pt", weights_only=False)["identity"]})
fold_of = {i: j % 5 for j, i in enumerate(ids_train)}
est, Rl, Rt, split_of = {}, {}, {}, {}


def add(pred, only, split):
    for idx, ident in enumerate(pred["identity"]):
        if only is not None and not only(ident):
            continue
        split_of[ident] = split
        if pred["mu_cm"] is not None:
            d = est.setdefault(ident, {"mu": [], "sigma": [], "truth": pred["target_cm"][idx] / 100.0, "split": split})
            d["mu"].append(pred["mu_cm"][idx] / 100.0); d["sigma"].append(torch.exp(0.5 * pred["logvar_cm"][idx]) / 100.0)
        if pred["R_est"] is not None:
            Rl.setdefault(ident, []).append(pred["R_est"][idx]); Rt[ident] = pred["R_true"][idx]


for k in range(5):
    add(torch.load(D / f"pred_{MODE}_s{SEED}_fold{k}_train.pt", weights_only=False), lambda i, k=k: fold_of[i] == k, "train")
add(torch.load(D / f"pred_{MODE}_s{SEED}_full_val.pt", weights_only=False), None, "validation")
for d in est.values():
    d["mu"] = torch.stack(d["mu"]).float(); d["sigma"] = torch.stack(d["sigma"]).float()
R = {}
for ident, lst in Rl.items():
    M = torch.stack(lst).mean(0)                                            # [5,3,3] mean over sessions
    U, S, Vh = torch.linalg.svd(M)
    det = torch.det(U @ Vh)
    D3 = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], -1))
    R[ident] = (U @ D3 @ Vh).float()
if est:
    torch.save(est, OUT_EST)
torch.save({"R": R, "R_true": Rt, "split": split_of}, OUT_R)


def ang(A, B):
    c = ((A.transpose(-1, -2) @ B).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6)))


for s in ("train", "validation"):
    ids = [i for i in R if split_of[i] == s]
    if ids:
        e = torch.stack([ang(R[i], Rt[i]) for i in ids]); base = torch.stack([ang(torch.eye(3).expand(5, 3, 3), Rt[i]) for i in ids])
        print(f"{s:10s} identities {len(ids):3d}: orientation-offset error {float(e.mean()):.2f} deg (per sensor {e.mean(0).numpy().round(2)}) | no-information {float(base.mean()):.2f} deg")
    ids = [i for i in est if est[i]["split"] == s]
    if ids:
        errs = [((est[i]["mu"].mean(0) - est[i]["truth"]).reshape(8, 3).norm(dim=-1) * 100).mean() for i in ids]
        print(f"{s:10s} identities {len(ids):3d}: geometry (R2) error of the session-mean estimate {float(torch.stack(errs).mean()):.2f} cm")
