"""Stage B: audit, diagnose and FREEZE the calibration-error sampler (AMASS OOF identities only; no DIP/TC, no downstream metric is read).

Pre-specified (before looking at any diagnostics):  fit set = fold 0 (69 identities); check set = folds 1-4 (273 identities); the 49 validation identities are not used.
Main wearing = 8 aggregated sessions; 4- and 1-session samplers are sensitivity conditions.  Sampler = zero-mean Gaussian in the 15-d left-tangent space (5 sensors x 3 axes)
with a full covariance, truncated at the chi2_15 0.999 radius.  Acceptance of the main sampler (8 sessions) on the check set: |mean angle diff| <= 0.10 deg,
|P95 diff| <= 0.30 deg, per-sensor |mean angle diff| <= 0.15 deg.  A pooled all-342 sampler is saved as a SECONDARY artifact only.
usage: python scripts_run/line14_errsampler_build.py --out data_local/line14_errsampler_v1
"""
import argparse, glob, hashlib, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from mobileposer.articulate import math as M
from mobileposer.line14.error_sampler import SENSORS, CHI2_15_999, apply_error, draw_delta

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("data_local/line14_errsampler_v1")); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
D = Path("data_local/line15_cal_ids"); VERSION = "errsampler_v1"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
report = {"version": VERSION, "fit_fold": 0, "check_folds": [1, 2, 3, 4], "main_sessions": 8}
log = lambda *x: print(*x, flush=True)

# ------------------------------------------------------------------ 1. provenance
tr = torch.load(D / "train.pt", weights_only=False)
ids = sorted(set(tr["identity"])); fold_of = {i: j % 5 for j, i in enumerate(ids)}
man = {"scripts": {}, "preds": {}, "ckpts": {}, "folds": {}}
for k in range(5):
    J = set(json.load(open(f"data_local/folds/fold{k}.json")))
    assert J == {i for i in ids if fold_of[i] == k}, f"fold{k}.json differs from index%5 routing"
    man["folds"][f"fold{k}"] = {"n": len(J), "sha256": sha(f"data_local/folds/fold{k}.json")[:16]}
P = {}
for k in range(5):
    f = D / f"pred_rot_s42_fold{k}_train.pt"; P[k] = torch.load(f, weights_only=False)
    assert P[k]["key"] == tr["key"] and P[k]["identity"] == tr["identity"], "prediction order differs from train.pt"
    man["preds"][f.name] = sha(f)[:16]
man["preds"]["pred_rot_s42_full_val.pt"] = sha(D / "pred_rot_s42_full_val.pt")[:16]
man["ids_data"] = {"train.pt": sha(D / "train.pt")[:16], "val.pt": sha(D / "val.pt")[:16]}
log("fold routing and prediction order: OK")
for k in list(range(5)) + ["full"]:
    f = f"checkpoints/line14/C20/rot_s42_{'fold' + str(k) if k != 'full' else 'full'}.pt"
    ck = torch.load(f, map_location="cpu", weights_only=False)
    gru = {n: tuple(v.shape) for n, v in ck["model"].items() if n.startswith("gru.weight_ih_l")}
    nl = len([n for n in ck["model"] if n.startswith("gru.weight_ih_l") and "reverse" not in n])
    hidden = ck["model"]["gru.weight_ih_l0"].shape[0] // 3
    man["ckpts"][os.path.basename(f)] = {"sha256": sha(f)[:16], "mode": ck.get("mode"), "gru_layers": nl, "gru_hidden": hidden, "has_geo_head": any(n.startswith("geo.") for n in ck["model"]), "has_rot_head": any(n.startswith("rot.") for n in ck["model"])}
    log(os.path.basename(f), man["ckpts"][os.path.basename(f)])
# training-set sizes of the fold models (the estimator drops every virtual subject whose identity is in the held-out fold)
ntr = {}
for nm in ["fold0", "fold1", "fold2", "fold3", "fold4", "full"]:
    j = json.load(open(f"results/line14/C20/rot_s42_{nm}.json"))
    assert j["status"] == "formal_validation_only" and j["test_accessed"] is False and j["scale_range"] == [0.5, 1.2] and j["rot_deg_prior"] == 5.0
    ntr[nm] = {"n_train": j["n_train"], "best_epoch": j["best_epoch"], "val_rot_deg_session_level": round(j["val"]["rot_deg"], 3)}
for k in range(5):
    ntr[f"fold{k}"]["removed_vs_full"] = ntr["full"]["n_train"] - ntr[f"fold{k}"]["n_train"]; ntr[f"fold{k}"]["identities_in_fold"] = man["folds"][f"fold{k}"]["n"]
man["training_sets"] = ntr
log("training samples per rot model (removed_vs_full ~ share of virtual subjects whose beta/identity is in the held-out fold):", ntr)
man["scripts"] = {n: sha(f"scripts_run/{n}")[:16] for n in ("line14_link_estimator.py", "line14_assemble_link.py", "line14_errsampler_build.py")}
man["scripts"]["error_sampler.py"] = sha("mobileposer/line14/error_sampler.py")[:16]

# ------------------------------------------------------------------ 2. error vectors (identity-level, aggregated like the downstream pipeline)
def log_so3(R):
    """float64 log map R[...,3,3] -> axis*angle [...,3]; stable for tiny angles (atan2 of the antisymmetric part), unlike acos(trace)."""
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1) / 2
    n = v.norm(dim=-1, keepdim=True); c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).unsqueeze(-1)
    th = torch.atan2(n, c)
    return torch.where(n > 1e-12, v / n.clamp_min(1e-12) * th, v)


def project(Mm):
    U, S, Vh = torch.linalg.svd(Mm); d = torch.det(U @ Vh)
    return U @ torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1)) @ Vh

def agg_err(p, idx):
    Rm = project(p["R_est"][idx].double().mean(0)); Rt = p["R_true"][idx[0]].double()
    return log_so3(Rm @ Rt.transpose(-1, -2))                                                                  # radians, left tangent, float64

E = {1: [], 4: [], 8: []}; meta = {1: [], 4: [], 8: []}
for i in ids:
    k = fold_of[i]; p = P[k]
    idx = sorted([n for n, x in enumerate(p["identity"]) if x == i], key=lambda n: int(p["key"][n].split("::")[-1]))
    assert len(idx) == 8 and all(torch.allclose(p["R_true"][idx[0]], p["R_true"][n]) for n in idx)
    for ns in (1, 4, 8):
        for g in range(8 // ns):
            E[ns].append(agg_err(p, idx[g * ns:(g + 1) * ns]).reshape(15)); meta[ns].append((i, k))
E = {ns: torch.stack(v) for ns, v in E.items()}
F = {ns: torch.tensor([m[1] for m in meta[ns]]) for ns in E}; IDX = {ns: [m[0] for m in meta[ns]] for ns in E}
deg = lambda x: torch.rad2deg(x)
ang = lambda v: v.reshape(-1, 5, 3).norm(dim=-1)                                       # [N,5] radians
log({ns: tuple(E[ns].shape) for ns in E})

# ------------------------------------------------------------------ 3. diagnostics
def sel(ns, folds):
    m = torch.zeros(len(F[ns]), dtype=torch.bool)
    for f in folds: m |= (F[ns] == f)
    return E[ns][m]
def cov0(X): return (X.T @ X) / len(X)                                                 # zero-mean second moment
def corr(C): d = C.diag().sqrt().clamp_min(1e-12); return C / d[:, None] / d[None, :]
diag = {}
for ns in (8, 4, 1):
    fit, chk = sel(ns, [0]), sel(ns, [1, 2, 3, 4])
    d = {"n_fit": len(fit), "n_check": len(chk)}
    d["mean_angle_deg"] = {"fit": float(deg(ang(fit)).mean()), "check": float(deg(ang(chk)).mean())}
    d["per_fold_mean_angle_deg"] = {f"fold{f}": float(deg(ang(sel(ns, [f]))).mean()) for f in range(5)}
    # bias
    for nm, X in (("fit", fit), ("check", chk)):
        mu = deg(X.mean(0)); se = deg(X.std(0)) / np.sqrt(len(X))
        d[f"bias_{nm}_max_abs_deg"] = float(mu.abs().max()); d[f"bias_{nm}_max_abs_tstat"] = float((mu / se).abs().max())
    d["bias_sign_agreement_fit_vs_check"] = float((torch.sign(fit.mean(0)) == torch.sign(chk.mean(0))).float().mean())
    # covariance / correlation
    Cf, Cc = cov0(fit), cov0(chk)
    Rf, Rc = corr(Cf), corr(Cc)
    off = ~torch.eye(15, dtype=torch.bool)
    cross = torch.ones(15, 15, dtype=torch.bool)
    for s in range(5): cross[s * 3:(s + 1) * 3, s * 3:(s + 1) * 3] = False                # between-sensor blocks
    d["corr_fro_rel_diff_fit_vs_check"] = float((Rf - Rc).norm() / Rc.norm())
    d["cov_fro_rel_diff_fit_vs_check"] = float((Cf - Cc).norm() / Cc.norm())
    d["mean_abs_between_sensor_corr"] = {"fit": float(Rf[cross].abs().mean()), "check": float(Rc[cross].abs().mean())}
    d["max_abs_between_sensor_corr"] = {"fit": float(Rf[cross].abs().max()), "check": float(Rc[cross].abs().max())}
    top = torch.topk(Rf.abs().masked_fill(~cross, 0).flatten(), 6).indices
    d["top_between_sensor_pairs_fit_vs_check"] = [(f"{SENSORS[int(t) // 15 // 3]}.{int(t) // 15 % 3}-{SENSORS[int(t) % 15 // 3]}.{int(t) % 3}", round(float(Rf.flatten()[t]), 2), round(float(Rc.flatten()[t]), 2)) for t in top[::2]]
    ef, ec = torch.linalg.eigvalsh(Cf).flip(0), torch.linalg.eigvalsh(Cc).flip(0)
    d["sqrt_eigvals_deg_fit"] = [round(float(deg(x.sqrt())), 3) for x in ef[:6]]; d["sqrt_eigvals_deg_check"] = [round(float(deg(x.sqrt())), 3) for x in ec[:6]]
    d["sqrt_eigvals_deg_smallest_fit_check"] = [round(float(deg(ef[-1].clamp_min(0).sqrt())), 4), round(float(deg(ec[-1].clamp_min(0).sqrt())), 4)]
    # calibration of the fitted covariance on the check set (Mahalanobis)
    w, V = torch.linalg.eigh(Cf); w = w.clamp_min(1e-10)
    z = (chk @ V) / w.sqrt(); d2 = (z ** 2).sum(1)
    q = {p: float(np.quantile(d2.numpy(), p)) for p in (0.5, 0.95, 0.99)}
    from math import comb  # noqa
    chi2 = {0.5: 14.339, 0.95: 24.996, 0.99: 30.578}
    d["mahalanobis_check_mean_over_15"] = float(d2.mean() / 15); d["mahalanobis_check_quantiles_vs_chi2_15"] = {str(p): (round(q[p], 1), chi2[p]) for p in q}
    d["frac_check_beyond_chi2_0.999"] = float((d2 > CHI2_15_999).float().mean())
    d["per_sensor_kurtosis_of_angle_check"] = [round(float(((a_ - a_.mean()) ** 4).mean() / a_.var(unbiased=False) ** 2), 2) for a_ in ang(chk).T]
    diag[ns] = d
# variance decomposition: persistent (identity-level) vs session-specific, from the raw per-session errors
sess = {i: [] for i in ids}
for i in ids:
    k = fold_of[i]; p = P[k]
    idx = sorted([n for n, x in enumerate(p["identity"]) if x == i], key=lambda n: int(p["key"][n].split("::")[-1]))
    for n in idx: sess[i].append(agg_err(p, [n]).reshape(15))
S = torch.stack([torch.stack(sess[i]) for i in ids])                                  # [342,8,15]
within = S.var(1, unbiased=True).mean(0); between = S.mean(1).var(0, unbiased=True) - within / 8
report["variance_decomposition"] = {"per_sensor_persistent_share": [round(float((between[s * 3:(s + 1) * 3].sum() / (between[s * 3:(s + 1) * 3].sum() + within[s * 3:(s + 1) * 3].sum())).clamp(0, 1)), 2) for s in range(5)],
                                    "note": "share of the single-session error variance that is identity-persistent (does not average out over sessions)"}
report["diagnostics"] = diag
for ns in (8, 4, 1):
    d = diag[ns]
    log(f"\n=== ns={ns}: fit n={d['n_fit']} check n={d['n_check']} | mean angle fit/check {d['mean_angle_deg']['fit']:.3f}/{d['mean_angle_deg']['check']:.3f} deg | per-fold {np.round(list(d['per_fold_mean_angle_deg'].values()), 3)}")
    log(f"   bias: max|mean| fit {d['bias_fit_max_abs_deg']:.2f} deg (max |t| {d['bias_fit_max_abs_tstat']:.1f}), check {d['bias_check_max_abs_deg']:.2f} deg (max |t| {d['bias_check_max_abs_tstat']:.1f}); sign agreement fit vs check {d['bias_sign_agreement_fit_vs_check']:.2f}")
    log(f"   cov rel diff {d['cov_fro_rel_diff_fit_vs_check']:.2f} | corr rel diff {d['corr_fro_rel_diff_fit_vs_check']:.2f} | between-sensor |corr| mean fit/check {d['mean_abs_between_sensor_corr']['fit']:.3f}/{d['mean_abs_between_sensor_corr']['check']:.3f}, max {d['max_abs_between_sensor_corr']['fit']:.2f}/{d['max_abs_between_sensor_corr']['check']:.2f}")
    log(f"   top between-sensor pairs (name, fit corr, check corr): {d['top_between_sensor_pairs_fit_vs_check']}")
    log(f"   sqrt eig (deg) fit {d['sqrt_eigvals_deg_fit']} | check {d['sqrt_eigvals_deg_check']} | smallest {d['sqrt_eigvals_deg_smallest_fit_check']}")
    log(f"   Mahalanobis on check (fold-0 covariance): mean/15 {d['mahalanobis_check_mean_over_15']:.2f}; quantiles (obs, chi2) {d['mahalanobis_check_quantiles_vs_chi2_15']}; beyond chi2 .999: {d['frac_check_beyond_chi2_0.999']:.4f}; angle kurtosis/sensor {d['per_sensor_kurtosis_of_angle_check']}")
log("\npersistent (identity-level) share of single-session error variance per sensor:", report["variance_decomposition"]["per_sensor_persistent_share"])

# ------------------------------------------------------------------ 4. build samplers, validate on the check set
def make(X, jitter=1e-9):
    C = cov0(X).double(); w, V = torch.linalg.eigh(C); w = w.clamp_min(jitter)
    return {"cov": C, "L": (V * w.sqrt()).float(), "n_fit": len(X)}
sampler = {"version": VERSION, "sensors": SENSORS, "convention": "left tangent: R_est = Exp(delta) @ R_oracle, delta = Log(R_est R_true^T) radians, per sensor [lw,rw,ls,rs,pelvis]; stored R_off convention: measured = R_bone @ R_off; zero-mean Gaussian, truncated at chi2_15 0.999",
           "fit_fold": 0, "fit_identities_sha256": hashlib.sha256("|".join(sorted(i for i in ids if fold_of[i] == 0)).encode()).hexdigest()[:16], "seed_policy": "sha256(key|mc|version|ns)", "ns": {}}
secondary = {**{k: v for k, v in sampler.items() if k != "ns"}, "version": VERSION + "_pooled342", "fit_fold": "all", "ns": {}}
for ns in (8, 4, 1):
    sampler["ns"][ns] = make(sel(ns, [0])); secondary["ns"][ns] = make(sel(ns, [0, 1, 2, 3, 4]))
gen = torch.Generator().manual_seed(a.seed)
val = {}
CRIT = {"mean_angle_deg": 0.10, "p95_deg": 0.30, "per_sensor_mean_deg": 0.15}
for ns in (8, 4, 1):
    chk = sel(ns, [1, 2, 3, 4]); emp = deg(ang(chk))                                    # [N,5]
    # draw through the public function with synthetic wearing keys (checks determinism and truncation)
    Ds = torch.stack([draw_delta(sampler, f"val_wearing_{n}", 0, ns).reshape(15) for n in range(20000)]).double()
    smp = deg(ang(Ds))
    row = {"empirical_mean": float(emp.mean()), "sampled_mean": float(smp.mean()), "empirical_p95_pooled": float(np.quantile(emp.numpy(), .95)), "sampled_p95_pooled": float(np.quantile(smp.numpy(), .95)),
           "empirical_p99_pooled": float(np.quantile(emp.numpy(), .99)), "sampled_p99_pooled": float(np.quantile(smp.numpy(), .99)),
           "per_sensor_mean_emp": [round(float(x), 3) for x in emp.mean(0)], "per_sensor_mean_smp": [round(float(x), 3) for x in smp.mean(0)],
           "sampled_cov_rel_err_vs_fit": float((cov0(Ds) - sampler["ns"][ns]["cov"]).norm() / sampler["ns"][ns]["cov"].norm())}
    row["diff_mean"] = row["sampled_mean"] - row["empirical_mean"]; row["diff_p95"] = row["sampled_p95_pooled"] - row["empirical_p95_pooled"]
    row["max_abs_diff_per_sensor_mean"] = float(np.max(np.abs(np.array(row["per_sensor_mean_smp"]) - np.array(row["per_sensor_mean_emp"]))))
    row["accepted"] = bool(abs(row["diff_mean"]) <= CRIT["mean_angle_deg"] and abs(row["diff_p95"]) <= CRIT["p95_deg"] and row["max_abs_diff_per_sensor_mean"] <= CRIT["per_sensor_mean_deg"])
    # leave-one-fold-out: fit on the other 4 folds, compare with the held-out fold (is a single fold-0 fit representative?)
    lofo = []
    for f in range(5):
        Cl = cov0(sel(ns, [x for x in range(5) if x != f])).double(); w, V = torch.linalg.eigh(Cl); Lm = (V * w.clamp_min(1e-9).sqrt()).float()
        Zs = torch.randn(20000, 15, generator=gen); Zs = Zs[(Zs ** 2).sum(1) <= CHI2_15_999]
        sm = deg(ang(Zs @ Lm.T)); ho = deg(ang(sel(ns, [f])))
        lofo.append((f, round(float(sm.mean() - ho.mean()), 3), round(float(np.quantile(sm.numpy(), .95) - np.quantile(ho.numpy(), .95)), 3)))
    row["leave_one_fold_out_(fold, d_mean, d_p95)"] = lofo
    val[ns] = row
    log(f"\nsampler ns={ns} vs check set: mean {row['sampled_mean']:.3f} vs {row['empirical_mean']:.3f} (diff {row['diff_mean']:+.3f}) | pooled P95 {row['sampled_p95_pooled']:.2f} vs {row['empirical_p95_pooled']:.2f} (diff {row['diff_p95']:+.2f}) | P99 {row['sampled_p99_pooled']:.2f} vs {row['empirical_p99_pooled']:.2f}")
    log(f"   per-sensor mean (sampled) {row['per_sensor_mean_smp']} (empirical) {row['per_sensor_mean_emp']} | max diff {row['max_abs_diff_per_sensor_mean']:.3f} | sampled-cov rel err {row['sampled_cov_rel_err_vs_fit']:.3f} | ACCEPTED: {row['accepted']}")
    log(f"   leave-one-fold-out (fold, d_mean, d_p95): {lofo}")
report["validation"] = val; report["acceptance_criteria"] = CRIT
# determinism + injection sanity
d1 = draw_delta(sampler, "id_x", 3, 8); d2 = draw_delta(sampler, "id_x", 3, 8); d3 = draw_delta(sampler, "id_x", 4, 8)
assert torch.equal(d1, d2) and not torch.equal(d1, d3)
Rt = P[0]["R_true"][0].float()
Re = apply_error(Rt, d1)
back = log_so3(Re.double() @ Rt.double().transpose(-1, -2))
assert torch.allclose(back.float(), d1, atol=1e-5), "apply_error does not realise the left-tangent error"
assert torch.allclose(Re @ Re.transpose(-1, -2), torch.eye(3).expand(5, 3, 3), atol=1e-5)
log("\ndeterminism / SO(3) / convention round-trip: OK")

# ------------------------------------------------------------------ 5. freeze
torch.save(sampler, a.out / "sampler.pt"); torch.save(secondary, a.out / "sampler_pooled342_secondary.pt")
man["sampler_sha256"] = {"sampler.pt": sha(a.out / "sampler.pt")[:16], "sampler_pooled342_secondary.pt": sha(a.out / "sampler_pooled342_secondary.pt")[:16]}
report["manifest"] = man
json.dump(report, open(a.out / "report.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
log("\nfrozen:", a.out, man["sampler_sha256"])
