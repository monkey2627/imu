"""Stage B, step 2: choose the covariance STRUCTURE of the calibration-error sampler by leave-one-fold-out (LOFO) held-out Gaussian log-likelihood on the AMASS OOF
error vectors (no DIP/TC, no downstream metric).  Why: sampler v1 (full 15x15 covariance fitted on fold 0 only, n=69) matched the marginal mean/P95 but was badly
miscalibrated jointly on the other folds (Mahalanobis mean/15 = 1.9, 22.7% beyond the chi2_15 0.999 radius).

Declared rule (before running):  candidates = iso (per-sensor isotropic), diag (15 variances), block (5 independent 3x3 blocks), shrink (0.5 full + 0.5 block), full (15x15).
Score = mean held-out log-likelihood per wearing at ns=8 (5 folds, each fitted on the other four).  Pick the best; if a simpler candidate is within one standard error
(over folds) of the best, pick the simplest such candidate.  The chosen structure is applied to every ns, refitted on all 342 identities, validated by LOFO
(marginal mean / P95 per held-out fold, Mahalanobis calibration) and frozen as errsampler_v2.   usage: python scripts_run/line14_errsampler_select.py
"""
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from mobileposer.line14.error_sampler import SENSORS, CHI2_15_999, draw_delta

D = Path("data_local/line15_cal_ids"); OUT = Path("data_local/line14_errsampler_v2"); OUT.mkdir(parents=True, exist_ok=True)
VERSION = "errsampler_v2"; log = lambda *x: print(*x, flush=True); deg = torch.rad2deg
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
tr = torch.load(D / "train.pt", weights_only=False); ids = sorted(set(tr["identity"])); fold_of = {i: j % 5 for j, i in enumerate(ids)}
P = {k: torch.load(D / f"pred_rot_s42_fold{k}_train.pt", weights_only=False) for k in range(5)}


def log_so3(R):
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1) / 2
    n = v.norm(dim=-1, keepdim=True); c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).unsqueeze(-1)
    return torch.where(n > 1e-12, v / n.clamp_min(1e-12) * torch.atan2(n, c), v)


def project(Mm):
    U, S, Vh = torch.linalg.svd(Mm); d = torch.det(U @ Vh)
    return U @ torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1)) @ Vh


E = {1: [], 4: [], 8: []}; F = {1: [], 4: [], 8: []}
for i in ids:
    k = fold_of[i]; p = P[k]
    idx = sorted([n for n, x in enumerate(p["identity"]) if x == i], key=lambda n: int(p["key"][n].split("::")[-1]))
    Rt = p["R_true"][idx[0]].double()
    for ns in (1, 4, 8):
        for g in range(8 // ns):
            Rm = project(p["R_est"][idx[g * ns:(g + 1) * ns]].double().mean(0))
            E[ns].append(log_so3(Rm @ Rt.transpose(-1, -2)).reshape(15)); F[ns].append(k)
E = {ns: torch.stack(v) for ns, v in E.items()}; F = {ns: torch.tensor(v) for ns, v in F.items()}
ang = lambda v: v.reshape(-1, 5, 3).norm(dim=-1)
FLOOR = 1e-9                                                                  # rad^2 (3e-5 rad ~ 0.002 deg)


def second_moment(X): return (X.T @ X) / len(X)


def structure(C, kind):
    if kind == "full": return C
    B = torch.zeros_like(C)
    for s in range(5): B[s * 3:(s + 1) * 3, s * 3:(s + 1) * 3] = C[s * 3:(s + 1) * 3, s * 3:(s + 1) * 3]
    if kind == "block": return B
    if kind == "shrink": return 0.5 * C + 0.5 * B
    if kind == "diag": return torch.diag(torch.diag(C))
    if kind == "iso":
        d = torch.zeros(15, dtype=C.dtype)
        for s in range(5): d[s * 3:(s + 1) * 3] = C.diag()[s * 3:(s + 1) * 3].mean()
        return torch.diag(d)
    raise ValueError(kind)


N_PARAMS = {"iso": 5, "diag": 15, "block": 30, "shrink": 120, "full": 120}
KINDS = ["iso", "diag", "block", "shrink", "full"]


def factor(C):
    w, V = torch.linalg.eigh(C); w = w.clamp_min(FLOOR); return w, V


def heldout(C, X):
    w, V = factor(C); z = (X @ V) / w.sqrt(); d2 = (z ** 2).sum(1)
    ll = -0.5 * (d2 + torch.log(w).sum() + 15 * np.log(2 * np.pi))
    return ll, d2


res = {}
for ns in (8, 4, 1):
    res[ns] = {}
    for kind in KINDS:
        lls, d2m, tail = [], [], []
        for f in range(5):
            Xtr, Xte = E[ns][F[ns] != f], E[ns][F[ns] == f]
            ll, d2 = heldout(structure(second_moment(Xtr), kind), Xte)
            lls.append(float(ll.mean())); d2m.append(float(d2.mean() / 15)); tail.append(float((d2 > CHI2_15_999).float().mean()))
        res[ns][kind] = {"ll_mean": float(np.mean(lls)), "ll_se": float(np.std(lls, ddof=1) / np.sqrt(5)), "ll_per_fold": [round(x, 2) for x in lls],
                         "mahal_mean_over_15": float(np.mean(d2m)), "frac_beyond_chi2_999": float(np.mean(tail))}
    # reference: the frozen v1 recipe (full covariance, fold 0 only) evaluated on folds 1-4
    ll, d2 = heldout(structure(second_moment(E[ns][F[ns] == 0]), "full"), E[ns][F[ns] != 0])
    res[ns]["v1_full_fold0_on_folds1-4"] = {"ll_mean": float(ll.mean()), "mahal_mean_over_15": float(d2.mean() / 15), "frac_beyond_chi2_999": float((d2 > CHI2_15_999).float().mean())}
    log(f"\n=== LOFO held-out log-likelihood per wearing (nats; higher is better), ns={ns}")
    for kind in KINDS + ["v1_full_fold0_on_folds1-4"]:
        r = res[ns][kind]
        log(f"  {kind:26s} ll {r['ll_mean']:8.2f}" + (f" ± {r['ll_se']:.2f}" if "ll_se" in r else "          ") + f" | Mahalanobis mean/15 {r['mahal_mean_over_15']:.2f} | beyond chi2 .999: {r['frac_beyond_chi2_999']:.4f}")

# ---- declared selection rule at ns = 8
best = max(KINDS, key=lambda k: res[8][k]["ll_mean"]); thr = res[8][best]["ll_mean"] - res[8][best]["ll_se"]
chosen = min([k for k in KINDS if res[8][k]["ll_mean"] >= thr], key=lambda k: N_PARAMS[k] + 0.01 * KINDS.index(k))
log(f"\nbest by held-out log-lik: {best}; within one SE of the best: {[k for k in KINDS if res[8][k]['ll_mean'] >= thr]}; CHOSEN (simplest within 1 SE): {chosen}")

# ---- fit chosen structure on all 342 identities, validate by LOFO, freeze
sampler = {"version": VERSION, "sensors": SENSORS, "structure": chosen,
           "convention": "left tangent: R_est = Exp(delta) @ R_oracle, delta = Log(R_est R_true^T) radians, per sensor [lw,rw,ls,rs,pelvis]; stored R_off convention: measured = R_bone @ R_off; zero-mean Gaussian, truncated at chi2_15 0.999",
           "fit_fold": "all (342 OOF identities)", "seed_policy": "sha256(key|mc|version|ns)", "ns": {}}
val = {}
gen = torch.Generator().manual_seed(42)
for ns in (8, 4, 1):
    C = structure(second_moment(E[ns]), chosen).double(); w, V = factor(C)
    sampler["ns"][ns] = {"cov": C, "L": (V * w.sqrt()).float(), "n_fit": len(E[ns])}
    rows = []
    for f in range(5):                                                         # LOFO marginal check of the chosen structure
        Cl = structure(second_moment(E[ns][F[ns] != f]), chosen).double(); wl, Vl = factor(Cl); Lm = (Vl * wl.sqrt()).float()
        Z = torch.randn(40000, 15, generator=gen); Z = Z[(Z ** 2).sum(1) <= CHI2_15_999]
        sm, ho = deg(ang((Z @ Lm.T).double())), deg(ang(E[ns][F[ns] == f]))
        rows.append({"fold": f, "d_mean": round(float(sm.mean() - ho.mean()), 3), "d_p95": round(float(np.quantile(sm.numpy(), .95) - np.quantile(ho.numpy(), .95)), 3),
                     "d_p99": round(float(np.quantile(sm.numpy(), .99) - np.quantile(ho.numpy(), .99)), 3),
                     "max_abs_d_sensor_mean": round(float((sm.mean(0) - ho.mean(0)).abs().max()), 3)})
    Ds = torch.stack([draw_delta(sampler, f"val_wearing_{n}", 0, ns).reshape(15) for n in range(20000)]).double()
    smp, emp = deg(ang(Ds)), deg(ang(E[ns]))
    val[ns] = {"lofo": rows, "pooled_all342": {"mean_sampled": float(smp.mean()), "mean_empirical": float(emp.mean()), "p95_sampled": float(np.quantile(smp.numpy(), .95)), "p95_empirical": float(np.quantile(emp.numpy(), .95)),
                                             "p99_sampled": float(np.quantile(smp.numpy(), .99)), "p99_empirical": float(np.quantile(emp.numpy(), .99)),
                                             "sensor_mean_sampled": [round(float(x), 3) for x in smp.mean(0)], "sensor_mean_empirical": [round(float(x), 3) for x in emp.mean(0)]}}
    pp = val[ns]["pooled_all342"]
    log(f"\nns={ns} ({chosen}) pooled-342: mean {pp['mean_sampled']:.3f} vs {pp['mean_empirical']:.3f} | P95 {pp['p95_sampled']:.2f} vs {pp['p95_empirical']:.2f} | P99 {pp['p99_sampled']:.2f} vs {pp['p99_empirical']:.2f}")
    log(f"   per-sensor mean sampled {pp['sensor_mean_sampled']} empirical {pp['sensor_mean_empirical']}")
    log(f"   LOFO (fit on 4 folds, compare with the held-out fold): " + "; ".join(f"f{r['fold']}: dmean {r['d_mean']:+.3f} dP95 {r['d_p95']:+.3f} dP99 {r['d_p99']:+.3f} max|dsensor| {r['max_abs_d_sensor_mean']:.3f}" for r in rows))
d1, d2_ = draw_delta(sampler, "id_x", 3, 8), draw_delta(sampler, "id_x", 3, 8); assert torch.equal(d1, d2_) and not torch.equal(d1, draw_delta(sampler, "id_x", 4, 8))
torch.save(sampler, OUT / "sampler.pt")
man = {"sampler.pt": sha(OUT / "sampler.pt")[:16], "builder_script": sha("scripts_run/line14_errsampler_select.py")[:16], "error_sampler.py": sha("mobileposer/line14/error_sampler.py")[:16],
       "v1_manifest": json.load(open("data_local/line14_errsampler_v1/report.json"))["manifest"], "ids_sha": hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]}
json.dump({"selection": res, "chosen": chosen, "best_by_ll": best, "validation": val, "manifest": man}, open(OUT / "report.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
log("\nfrozen v2:", OUT, man["sampler.pt"])
