"""Assemble cross-fitted R2 estimates with *two calibration qualities* (E13).
usage: python scripts_run/line14_assemble_est2.py GOOD_DIR BAD_DIR OUT
For each downstream identity, sessions of both qualities are kept ('q' = 0 good / 1 bad).  Train identities use the fold model that
did not see them; validation identities use the full model."""
import sys
from pathlib import Path
import numpy as np, torch

GOOD, BAD, OUT = [Path(x) for x in sys.argv[1:4]]
PFX = sys.argv[4] if len(sys.argv) > 4 else ""
ids_train = sorted({i for i in torch.load(GOOD / "train.pt", weights_only=False)["identity"]})
fold_of = {i: j % 5 for j, i in enumerate(ids_train)}
out = {}


def add(pred, q, only=None, split="train"):
    for idx, ident in enumerate(pred["identity"]):
        if only is not None and not only(ident): continue
        d = out.setdefault(ident, {"mu": [], "sigma": [], "q": [], "truth": pred["target_cm"][idx] / 100.0, "split": split})
        d["mu"].append(pred["mu_cm"][idx] / 100.0); d["sigma"].append(torch.exp(0.5 * pred["logvar_cm"][idx]) / 100.0); d["q"].append(q)


for q, D in enumerate((GOOD, BAD)):
    for k in range(5):
        add(torch.load(D / f"pred_{PFX}fold{k}_train.pt", weights_only=False), q, only=lambda i, k=k: fold_of[i] == k, split="train")
    add(torch.load(D / f"pred_{PFX}full_val.pt", weights_only=False), q, split="validation")
for d in out.values():
    d["mu"] = torch.stack(d["mu"]).float(); d["sigma"] = torch.stack(d["sigma"]).float(); d["q"] = torch.tensor(d["q"])
torch.save(out, OUT)
for split in ("train", "validation"):
    for q, name in ((0, "good"), (1, "bad")):
        e, s = [], []
        for d in out.values():
            if d["split"] != split: continue
            m = d["q"] == q
            e.append(((d["mu"][m] - d["truth"]).reshape(-1, 8, 3).norm(dim=-1)).mean() * 100)
            s.append(d["sigma"][m].reshape(-1, 8, 3).norm(dim=-1).mean() * 100)
        print(f"{split:10s} {name:4s}: R2 error {float(np.mean([float(x) for x in e])):.2f} cm | mean sigma-hat {float(np.mean([float(x) for x in s])):.2f} cm")
