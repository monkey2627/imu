"""Assemble cross-fitted R2 estimates for the downstream identities (E8).
train identities: from the fold model that did NOT see them in training (fold = index % 5 of the sorted train identities);
validation identities: from the full model.  Output: dict identity -> {mu [S,24] m, sigma [S,24] m, truth [24] m, split}"""
import json, sys
from pathlib import Path
import numpy as np, torch

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data_local/line14_cal_ids_dev10")
OUT = sys.argv[2] if len(sys.argv) > 2 else "data_local/line14_est_r2.pt"
PFX = sys.argv[3] if len(sys.argv) > 3 else ""
ids_train = sorted({i for i in torch.load(D / "train.pt", weights_only=False)["identity"]})
fold_of = {i: j % 5 for j, i in enumerate(ids_train)}
out = {}
def add(pred, only=None, split="train"):
    for idx, ident in enumerate(pred["identity"]):
        if only is not None and not only(ident): continue
        d = out.setdefault(ident, {"mu": [], "sigma": [], "truth": pred["target_cm"][idx] / 100.0, "split": split})
        d["mu"].append(pred["mu_cm"][idx] / 100.0); d["sigma"].append(torch.exp(0.5 * pred["logvar_cm"][idx]) / 100.0)
for k in range(5):
    add(torch.load(D / f"pred_{PFX}fold{k}_train.pt", weights_only=False), only=lambda i, k=k: fold_of[i] == k, split="train")
add(torch.load(D / f"pred_{PFX}full_val.pt", weights_only=False), split="validation")
for d in out.values():
    d["mu"] = torch.stack(d["mu"]).float(); d["sigma"] = torch.stack(d["sigma"]).float()
torch.save(out, OUT)
def stats(split):
    errs, base, sig = [], [], []
    tr_truth = torch.stack([d["truth"] for d in out.values() if d["split"] == "train"]); mean = tr_truth.mean(0)
    for d in out.values():
        if d["split"] != split: continue
        e = (d["mu"] - d["truth"]).reshape(-1, 8, 3).norm(dim=-1) * 100
        b = (mean - d["truth"]).reshape(8, 3).norm(dim=-1) * 100
        errs.append(e.mean()); base.append(b.mean()); sig.append(d["sigma"].reshape(-1, 8, 3).norm(dim=-1).mean() * 100)
    return float(np.mean(errs)), float(np.mean(base)), float(np.mean(sig)), len(errs)
for s in ("train", "validation"):
    e, b, sg, n = stats(s)
    print(f"{s:10s} identities {n:3d}: R2 estimate error {e:.2f} cm | train-mean baseline {b:.2f} cm | mean sigma-hat {sg:.2f} cm")
