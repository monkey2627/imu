"""Is the person-level bias of the (geometry-conditioned) Joints explained by body geometry the model does not use?
Eval-only, adult validation identities (49), one seed, frozen Joints variant.  For each identity: mean error vector of the 24 joints in the pelvis frame
(72-d).  Regress it on the identity's full TRUE geometry (24-d R2 + 23 bone lengths + beta10, standardised) with ridge, leave-one-identity-out,
and report the fraction of the cross-identity bias variance explained (LOO R^2) for several ridge strengths."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line9.models import Line9Joints, GeometryStats
from mobileposer.line14.geom import GeomProvider
from mobileposer.line14.models import FilmJoints
import line14_poser_chain as C

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--variant", default="hestp6"); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}"); EST = "data_local/line14_est_r2_b90.pt"
V = {"B0": (None, "checkpoints/line14/E5/r1_b0_s{s}.pt", None), "hestp6": ("hestp6", "checkpoints/line14/C9a/hestp6_oracle_film_s{s}.pt", EST)}
kind, path, est = V[a.variant]
ck = torch.load(path.format(s=a.seed), map_location="cpu", weights_only=False); st = GeometryStats(ck["geometry_mean"], ck["geometry_std"])
m = Line9Joints("b0", geom_dim=len(ck["geometry_mean"]), geometry_stats=st) if kind is None else FilmJoints(len(ck["geometry_mean"]), st)
m.load_state_dict(ck["model"]); m.to(dev).eval(); prov = GeomProvider(kind, 3.0, est, None) if kind else None
r2p = GeomProvider("r2", 3.0, None, None)
rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("validation",))
store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
val = PairedWindowDataset(rec["validation"], 125, store=store, file_cache_size=2, sequence_cache_size=64)
vd, _ = C.loader(val, 128, a.seed, False)
S, N, G = {}, {}, {}
with torch.no_grad():
    for b in vd:
        imu = b["target_imu"].to(dev); mask = b["target_mask"].to(dev); L = b["target_lengths"].tolist(); t = b["target_joints"][..., :3].to(dev)
        with torch.autocast(device_type="cuda"):
            p = (m(imu, L) if prov is None else m(imu, L, prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=a.seed).to(dev))).float()
        p = p.reshape(*t.shape); Rp = imu[..., 51:60].reshape(*imu.shape[:2], 3, 3)
        e = (Rp.transpose(-1, -2).unsqueeze(2) @ (p - t).unsqueeze(-1)).squeeze(-1) * mask[:, :, None, None]
        for k, ident in enumerate(b["identity"]):
            S[ident] = S.get(ident, 0) + e[k].sum(0).reshape(-1).cpu(); N[ident] = N.get(ident, 0) + int(mask[k].sum())
            if ident not in G:
                G[ident] = torch.cat([r2p.base(b["beta"][k], ident), b["bone_lengths"][k][1:], b["beta"][k]])
ids = sorted(S); Y = torch.stack([S[i] / N[i] for i in ids]).numpy(); X = torch.stack([G[i] for i in ids]).numpy()
X = (X - X.mean(0)) / (X.std(0) + 1e-9); Yc = Y - Y.mean(0)
tot = (Yc ** 2).sum()
print("identities", len(ids), "bias RMS per joint-coordinate (cm):", 100 * float(np.sqrt((Y ** 2).mean())), "variance across identities (cm^2):", 1e4 * float((Yc ** 2).mean()))
out = {}
for lam in (1, 10, 30, 100, 300):
    err = 0.0
    for i in range(len(ids)):
        tr = np.arange(len(ids)) != i
        mu = Y[tr].mean(0); W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ (Y[tr] - mu))
        err += ((Y[i] - mu - X[i] @ W) ** 2).sum()
    out[lam] = 1 - err / tot; print("ridge", lam, "LOO R2 of person-level bias explained by full true geometry:", round(float(out[lam]), 3))
Path("results/line14/C13").mkdir(parents=True, exist_ok=True); json.dump({str(k): float(v) for k, v in out.items()}, open(f"results/line14/C13/biasgeom_{a.variant}_s{a.seed}.json", "w"))

# does the residual person-level bias follow the AMASS sub-dataset the identity belongs to (mocap/fitting convention, motion repertoire)?
subs = np.array([i.split("/")[0] for i in ids]); err = 0.0; cnt = 0
for i in range(len(ids)):
    same = (subs == subs[i]) & (np.arange(len(ids)) != i)
    if same.sum() == 0: continue
    err += ((Y[i] - Y[same].mean(0)) ** 2).sum(); cnt += ((Y[i] - Y[np.arange(len(ids)) != i].mean(0)) ** 2).sum()
print("subsets:", {s: int((subs == s).sum()) for s in sorted(set(subs))})
print("LOO prediction of an identity's bias from the OTHER identities of the same sub-dataset: explained fraction (vs. predicting the global mean):", round(float(1 - err / cnt), 3), " (identities with >=1 sibling only)")
