"""Can calibration-derived features PREDICT a person's constant rotation (and position) bias?  Eval-only, adult validation identities (49), leave-one-identity-out ridge.
Target Y  : per-identity mean error vector in the pelvis frame of the end-to-end model (C5 e2e_film): 16 reduced-joint global-rotation errors (48-d, rad) and
            24 joint position errors (72-d, m).
Features  : (A) latent of the frozen R2 estimator (calibration session -> 256-d, mean over the identity's sessions), PCA-k
            (B) estimated PivotPrint pca6 (what the model already receives)  (C) full TRUE geometry (24 R2 + 23 bone lengths + beta10)
            (D) summary statistics of the raw calibration session (mean/std of the 135 phys features per phase group) PCA-k
Reported: LOO R^2 of the cross-identity variation of Y (>0 = the features carry information about the personal bias), ridge strengths scanned."""
import argparse, json
from pathlib import Path
import numpy as np, torch
import mobileposer.articulate as art
from mobileposer.articulate import math as AM
from mobileposer.config import paths
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line9.enc2 import phys_features
from mobileposer.line9.robust_calib import heading_normalize
from mobileposer.line14.geom import GeomProvider
import line14_poser_chain as C
import line14_e2e_code as E
from line14_estimate_r1 import R1Encoder

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}"); EST = "data_local/line14_est_r2_b90.pt"
# ---- model bias targets
st = torch.load(f"checkpoints/line14/C5/e2e_film_s{a.seed}.pt", map_location="cpu", weights_only=False)
prov = GeomProvider("estp6", 3.0, EST, None); jn = C.load_frozen(a.seed, dev)[1]; jn.load_state_dict(st["jnet"]); jn.eval()
poser = C.ChainPoser(132).to(dev); poser.load_state_dict(st["poser"]); poser.eval()
rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("validation",))
store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
val = PairedWindowDataset(rec["validation"], 125, store=store, file_cache_size=2, sequence_cache_size=64)
vd, _ = C.loader(val, 128, a.seed, False)
parent = art.model.ParametricModel(paths.smpl_file, device="cpu").parent
SR, NR, SP, GG = {}, {}, {}, {}
r2p = GeomProvider("r2", 3.0, None, None)
with torch.no_grad():
    for b in vd:
        imu = b["target_imu"].to(dev); mask = b["target_mask"].to(dev); L = b["target_lengths"].tolist(); t = b["target_joints"][..., :3].to(dev)
        ge = prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=a.seed).to(dev)
        with torch.autocast(device_type="cuda"):
            j = jn(imu, L, ge); p = poser(torch.cat((j.to(imu.dtype), imu), -1), L)
        j = j.float().reshape(*t.shape); p = p.float().reshape(*p.shape[:2], 16, 6)
        Rpel = imu[..., 51:60].reshape(*imu.shape[:2], 3, 3)
        ej = (Rpel.transpose(-1, -2).unsqueeze(2) @ (j - t).unsqueeze(-1)).squeeze(-1) * mask[:, :, None, None]
        Rp = AM.r6d_to_rotation_matrix(p.reshape(-1, 6)).reshape(*p.shape[:2], 16, 3, 3)
        y = C.global_targets(b["target_joints"][..., 3:].to(dev), parent).reshape(*p.shape[:2], 16, 6)
        Rg = AM.r6d_to_rotation_matrix(y.reshape(-1, 6)).reshape(*p.shape[:2], 16, 3, 3)
        ev = AM.rotation_matrix_to_axis_angle((Rp @ Rg.transpose(-1, -2)).reshape(-1, 3, 3)).reshape(*p.shape[:2], 16, 3)
        er = (Rpel.transpose(-1, -2).unsqueeze(2) @ ev.unsqueeze(-1)).squeeze(-1) * mask[:, :, None, None]
        for k, ident in enumerate(b["identity"]):
            n = int(mask[k].sum()); NR[ident] = NR.get(ident, 0) + n
            SR[ident] = SR.get(ident, 0) + er[k].sum(0).reshape(-1).cpu(); SP[ident] = SP.get(ident, 0) + ej[k].sum(0).reshape(-1).cpu()
            if ident not in GG:
                GG[ident] = torch.cat([r2p.base(b["beta"][k], ident), b["bone_lengths"][k][1:], b["beta"][k]])
ids = sorted(SR)
Yr = torch.stack([SR[i] / NR[i] for i in ids]).numpy(); Yp = torch.stack([SP[i] / NR[i] for i in ids]).numpy()
# ---- features
S_va, _, _ = E.load_sessions("val")                                  # (own normalisation; replaced below for the estimator latent)
est_ck = torch.load("checkpoints/line14/E17est/phys_r2_s42_b90_full.pt", map_location=dev, weights_only=False)
d = torch.load("data_local/line14_cal_ids_dyn13/val.pt", weights_only=False)
x = phys_features(heading_normalize(d["imu"].float()), dt=2 / 60.0)
xe = (x - est_ck["fm"].cpu()) / est_ck["fs"].cpu()
encr = R1Encoder(x.shape[-1], dim=24).to(dev); encr.load_state_dict(est_ck["model"]); encr.eval()
cap = {}
encr.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("z", o.detach().cpu()))
lat = []
with torch.no_grad():
    for i in range(0, len(xe), 64):
        encr(xe[i:i + 64].to(dev), d["phase_id"][i:i + 64].long().to(dev), torch.ones_like(d["phase_id"][i:i + 64], dtype=torch.bool).to(dev)); lat.append(cap["z"])
lat = torch.cat(lat)
by = {}
for k, ident in enumerate(d["identity"]): by.setdefault(ident, []).append(k)
FA = torch.stack([lat[by[i]].mean(0) for i in ids]).numpy()
FB = torch.stack([prov.base(None, i) for i in ids]).numpy()
FC = torch.stack([GG[i] for i in ids]).numpy()
ph = d["phase_id"].long(); xs = (x - x.mean((0, 1))) / x.std((0, 1)).clamp_min(1e-4)
def phase_stats(k):
    rows = []
    for ph_id in range(7):
        m = (ph[k] == ph_id)
        rows += [xs[k][m].mean(0), xs[k][m].std(0)] if m.any() else [torch.zeros(135), torch.zeros(135)]
    return torch.cat(rows)
FD = torch.stack([torch.stack([phase_stats(k) for k in by[i]]).mean(0) for i in ids]).numpy()

def pca(X, k):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); U, S, Vt = np.linalg.svd(X, full_matrices=False); return U[:, :k] * S[:k]
def loo(X, Y, lam):
    Yc = Y - Y.mean(0); tot = (Yc ** 2).sum(); err = 0.0
    for i in range(len(Y)):
        tr = np.arange(len(Y)) != i
        mu = Y[tr].mean(0); xm = X[tr].mean(0); xs_ = X[tr].std(0) + 1e-9; Xt = (X[tr] - xm) / xs_
        W = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (Y[tr] - mu))
        err += ((Y[i] - mu - ((X[i] - xm) / xs_) @ W) ** 2).sum()
    return 1 - err / tot
out = {}
for tname, Y in (("rotation_bias", Yr), ("position_bias", Yp)):
    print(f"== target {tname}: RMS bias {np.sqrt((Y ** 2).mean()):.4f}, cross-identity variance share {((Y - Y.mean(0)) ** 2).mean() / (Y ** 2).mean():.2f}")
    for fname, X in (("A est latent pca5", pca(FA, 5)), ("A est latent pca10", pca(FA, 10)), ("A est latent pca20", pca(FA, 20)), ("B est PivotPrint(6)", FB), ("C true geometry(33)", FC), ("D raw session stats pca10", pca(FD, 10)), ("D raw session stats pca20", pca(FD, 20))):
        r = {lam: float(loo(X, Y, lam)) for lam in (1, 10, 100, 1000)}
        out[f"{tname}|{fname}"] = r; print(f"   {fname:28s}", "  ".join(f"lam{l}: {v:+.3f}" for l, v in r.items()), flush=True)
Path("results/line14/C17").mkdir(parents=True, exist_ok=True); json.dump(out, open(f"results/line14/C17/predictbias_s{a.seed}.json", "w"), indent=1)
