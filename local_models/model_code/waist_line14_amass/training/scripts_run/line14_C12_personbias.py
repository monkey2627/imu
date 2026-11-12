"""Headroom of PERSON-LEVEL correction: how much of the Joints position error is a constant per-identity bias (in the pelvis frame)?
Eval-only (adult validation identities, one seed, frozen Joints). For each identity the windows are split into two interleaved halves;
the mean error vector per joint (pelvis frame) estimated on one half is subtracted from the other half. The reduction is an upper bound for
what any calibration that captures a constant personal offset (posture habit, bone-axis offset, ...) could give.
usage: python scripts_run/line14_C12_personbias.py --seed 42 --gpu 0 --variant {B0,estp6,hestp6}
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line9.models import Line9Joints, GeometryStats
from mobileposer.line14.geom import GeomProvider
from mobileposer.line14.models import FilmJoints
import line14_poser_chain as C

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--variant", default="estp6"); ap.add_argument("--imu-store", default="data_local/line14_store_mount3"); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}"); EST = "data_local/line14_est_r2_b90.pt"
V = {"B0clean": (None, "checkpoints/line14/b0_s{s}.pt", None), "B0": (None, "checkpoints/line14/E5/r1_b0_s{s}.pt", None), "estp6": ("estp6", "checkpoints/line14/E18/estp6_oracle_film_s{s}.pt", EST), "hestp6": ("hestp6", "checkpoints/line14/C9a/hestp6_oracle_film_s{s}.pt", EST)}
kind, path, est = V[a.variant]
ck = torch.load(path.format(s=a.seed), map_location="cpu", weights_only=False); st = GeometryStats(ck["geometry_mean"], ck["geometry_std"])
m = Line9Joints("b0", geom_dim=len(ck["geometry_mean"]), geometry_stats=st) if kind is None else FilmJoints(len(ck["geometry_mean"]), st)
m.load_state_dict(ck["model"]); m.to(dev).eval(); prov = GeomProvider(kind, 3.0, est, None) if kind else None
rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("validation",))
store = C.ChainStore(Path(a.imu_store), Path("data/line9_store_pose6d"))
val = PairedWindowDataset(rec["validation"], 125, store=store, file_cache_size=2, sequence_cache_size=64)
vd, _ = C.loader(val, 128, a.seed, False)
E = defaultdict(list)   # identity -> list of (window_sum_err_vec [24,3], count, window_sum_sq [24])
with torch.no_grad():
    for b in vd:
        imu = b["target_imu"].to(dev); mask = b["target_mask"].to(dev); L = b["target_lengths"].tolist(); t = b["target_joints"][..., :3].to(dev)
        with torch.autocast(device_type="cuda"):
            p = (m(imu, L) if prov is None else m(imu, L, prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=a.seed).to(dev))).float()
        p = p.reshape(*t.shape)
        Rp = imu[..., 15 + 36:15 + 45].reshape(*imu.shape[:2], 3, 3)                       # pelvis sensor orientation (slot 4)
        e = (Rp.transpose(-1, -2).unsqueeze(2) @ (p - t).unsqueeze(-1)).squeeze(-1)          # error in the pelvis frame [B,T,24,3]
        for k in range(len(e)):
            n = int(mask[k].sum()); ek = e[k, :n]
            E[b["identity"][k]].append((ek.sum(0).cpu(), n, (ek.norm(dim=-1) ).sum(0).cpu(), (ek ** 2).sum((0, 2)).cpu(), b["target_sequence"][k]))
tot = {"raw": 0.0, "after": 0.0, "n": 0, "bias_sq": 0.0, "sq": 0.0}
for ident, ws in E.items():
    if len(ws) < 4: continue
    seqs = sorted({w[4] for w in ws})
    if len(seqs) < 4: continue
    sa = set(seqs[0::2]); A, B = [w for w in ws if w[4] in sa], [w for w in ws if w[4] not in sa]      # split by SEQUENCE (held-out motions)
    for fit, ev in ((A, B), (B, A)):
        nf = sum(w[1] for w in fit); bias = sum(w[0] for w in fit) / nf                        # [24,3] mean error vector from the fit half
        ne = sum(w[1] for w in ev); mean_e = sum(w[0] for w in ev) / ne
        # error norm after subtracting the bias requires per-frame vectors; use  E||e-b||^2 = E||e||^2 - 2 b.mean_e + ||b||^2  (RMS-based, exact for squares)
        sq = sum(w[3] for w in ev) / ne                                                          # [24] mean squared norm per joint
        sq_after = sq - 2 * (bias * mean_e).sum(-1) + (bias ** 2).sum(-1)
        tot["raw"] += float(sq.mean()) * ne; tot["after"] += float(sq_after.mean()) * ne; tot["n"] += ne
r = {"split": "by_sequence", "variant": a.variant, "seed": a.seed, "rms_cm_raw": 100 * (tot["raw"] / tot["n"]) ** 0.5, "rms_cm_after_personal_bias": 100 * (tot["after"] / tot["n"]) ** 0.5}
r["mse_reduction_pct"] = 100 * (1 - tot["after"] / tot["raw"]); print(json.dumps(r))
Path("results/line14/C12").mkdir(parents=True, exist_ok=True); json.dump(r, open(f"results/line14/C12/personbias_seq_{a.variant}{'_clean' if 'shape' in a.imu_store else ''}_s{a.seed}.json", "w"))
