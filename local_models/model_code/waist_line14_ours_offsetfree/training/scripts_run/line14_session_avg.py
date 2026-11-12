"""Does averaging several calibration sessions of the same wear (same lever, different execution) reduce the R2 error?
Uses the cross-fitted predictions of E8estD (validation identities have 4 sessions, train identities 8)."""
import sys
import numpy as np, torch
D = sys.argv[1] if len(sys.argv) > 1 else "data_local/line14_cal_ids_dyn13"
from pathlib import Path
pred = torch.load(Path(D) / "pred_full_val.pt", weights_only=False)
ids = {}
for i, ident in enumerate(pred["identity"]):
    ids.setdefault(ident, []).append(i)
mu, tgt, lv = pred["mu_cm"], pred["target_cm"], pred["logvar_cm"]
def err(sel):
    out = []
    for ident, idx in ids.items():
        m = mu[idx[:sel]].mean(0); t = tgt[idx[0]]
        out.append((m - t).reshape(8, 3).norm(dim=-1).mean())
    return float(np.mean(out))
for k in (1, 2, 3, 4):
    print(f"average of {k} session(s): R2 error {err(k):.3f} cm")
# inverse-variance weighting
out = []
for ident, idx in ids.items():
    w = torch.exp(-lv[idx]); m = (mu[idx] * w).sum(0) / w.sum(0); out.append((m - tgt[idx[0]]).reshape(8, 3).norm(dim=-1).mean())
print("inverse-variance average of all sessions: %.3f cm" % float(np.mean(out)))
