"""Independent re-check of the stage-B OOF claims (read-only).  (1) fold routing vs data_local/folds/fold{k}.json, (2) per-fold files / shapes,
(3) identity-level orientation-offset error of the OOF rot estimator for 1/4/8 aggregated sessions, per-sensor, fold 0 vs folds 1-4, P95, signed per-axis bias."""
import json, hashlib
import numpy as np, torch
from pathlib import Path
from mobileposer.articulate import math as M

D = Path("data_local/line15_cal_ids")
tr = torch.load(D / "train.pt", weights_only=False)
ids = sorted(set(tr["identity"])); fold_of = {i: j % 5 for j, i in enumerate(ids)}
print("train identities", len(ids), "sessions", len(tr["identity"]))
# (1) fold json membership
for k in range(5):
    J = json.load(open(f"data_local/folds/fold{k}.json")); S = set(J)
    exp = {i for i in ids if fold_of[i] == k}
    print(f"fold{k}.json: {len(S)} ids | equals index%5 routing: {S == exp} | in train ids: {len(S & set(ids))} | extra not in train: {len(S - set(ids))} | missing: {len(exp - S)}")
# (2) per-fold prediction files
P = {}
for k in range(5):
    f = D / f"pred_rot_s42_fold{k}_train.pt"; p = torch.load(f, weights_only=False); P[k] = p
    print(f"{f.name}: sha256[:12]={hashlib.sha256(open(f, 'rb').read()).hexdigest()[:12]} identities={len(set(p['identity']))} sessions={len(p['identity'])} R_est={tuple(p['R_est'].shape)} same order as train.pt: {p['key'] == tr['key']}")


def project(Mm):
    U, S_, Vh = torch.linalg.svd(Mm); d = torch.det(U @ Vh)
    return U @ torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1)) @ Vh


def errs(sel_ids, nsess):
    out = []
    for i in sel_ids:
        k = fold_of[i]; p = P[k]; idx = [n for n, x in enumerate(p["identity"]) if x == i][:nsess]
        Rm = project(p["R_est"][idx].mean(0)); Rt = p["R_true"][idx[0]]
        e = M.rotation_matrix_to_axis_angle((Rm @ Rt.transpose(-1, -2)).reshape(-1, 3, 3)).reshape(5, 3)     # Log(R_hat R_true^T), world/left-multiplied form
        e2 = M.rotation_matrix_to_axis_angle((Rm.transpose(-1, -2) @ Rt).reshape(-1, 3, 3)).reshape(5, 3)    # Log(R_hat^T R_true), body form (convention check)
        out.append((torch.rad2deg(e), torch.rad2deg(e2)))
    return torch.stack([a for a, _ in out]), torch.stack([b for _, b in out])            # [N,5,3]


fit = [i for i in ids if fold_of[i] == 0]; chk = [i for i in ids if fold_of[i] != 0]
for ns in (1, 4, 8):
    for name, sel in (("fold0 fit", fit), ("folds1-4 check", chk)):
        e, e2 = errs(sel, ns); ang = e.norm(dim=-1); ang2 = e2.norm(dim=-1)
        print(f"sessions={ns} {name:15s} n={len(sel):3d} mean {float(ang.mean()):.3f} deg | P95 pooled {float(torch.quantile(ang.flatten(), .95)):.2f} | body-form mean {float(ang2.mean()):.3f} | per-sensor {ang.mean(0).numpy().round(2)}")
e, e2 = errs(fit, 8); ec, ec2 = errs(chk, 8)
print("\nsigned mean error (tangent space, deg) per sensor/axis, 8 sessions; fold0 vs folds1-4 (bias reproducibility); body-form in brackets")
for s_, nm in enumerate(["lw", "rw", "ls", "rs", "pelvis"]):
    print(f"  {nm:6s} fold0 {e[:, s_].mean(0).numpy().round(2)} folds1-4 {ec[:, s_].mean(0).numpy().round(2)} | body-form fold0 {e2[:, s_].mean(0).numpy().round(2)} folds1-4 {ec2[:, s_].mean(0).numpy().round(2)} | SD fold0 {e[:, s_].std(0).numpy().round(2)} SD chk {ec[:, s_].std(0).numpy().round(2)}")
# does the per-identity error depend on fold (OOF vs in-fold)?  in-fold = identity predicted by the model that EXCLUDED it (that is the OOF); compare with a model that TRAINED on it
leak = []
for i in chk[:80]:
    k_oof = fold_of[i]; k_other = (k_oof + 1) % 5
    for k, tag in ((k_oof, "oof"), (k_other, "in-training")):
        p = P[k]; idx = [n for n, x in enumerate(p["identity"]) if x == i]
        Rm = project(p["R_est"][idx].mean(0)); Rt = p["R_true"][idx[0]]
        leak.append((tag, float(torch.rad2deg(M.rotation_matrix_to_axis_angle((Rm @ Rt.transpose(-1, -2)).reshape(-1, 3, 3)).reshape(5, 3).norm(dim=-1)).mean())))
for tag in ("oof", "in-training"):
    print(f"identity-level error when predicted by the model that {'EXCLUDED' if tag == 'oof' else 'TRAINED ON'} the identity: {np.mean([v for t, v in leak if t == tag]):.3f} deg (80 identities)")
