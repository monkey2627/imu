"""Do PivotPrint / FiLM Joints improve bone DIRECTIONS (what a rotation-valued Poser needs) or only the overall scale/lengths?
Eval-only on the validation identities with the frozen B0 and FiLM(estp6) Joints of one seed: position error (cm), bone-direction error (deg)
and bone-length error (relative) per bone, plus the position error after removing the best global per-window scale (scale-invariant part).
usage: python scripts_run/line14_C6_dirdiag.py --seed 42 --gpu 0
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line14.geom import GeomProvider
import line14_poser_chain as C

BONES = {"hip->knee": [(1, 4), (2, 5)], "knee->ankle": [(4, 7), (5, 8)], "shoulder->elbow": [(16, 18), (17, 19)], "elbow->wrist": [(18, 20), (19, 21)]}
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}")
rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("validation",))
store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
val = PairedWindowDataset(rec["validation"], 125, store=store, file_cache_size=2, sequence_cache_size=64)
vd, _ = C.loader(val, 128, a.seed, False)
prov = GeomProvider("estp6", 3.0, "data_local/line14_est_r2_b90.pt")
b0, film = C.load_frozen(a.seed, dev)
acc = {s: {} for s in ("b0", "film")}
def add(s, k, v, n):
    d = acc[s].setdefault(k, [0.0, 0]); d[0] += v; d[1] += n
with torch.no_grad():
    for b in vd:
        imu = b["target_imu"].to(dev); m = b["target_mask"].to(dev); jt = b["target_joints"][..., :3].to(dev)
        ge = prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=a.seed).to(dev)
        for s in acc:
            p = C.joint_source(s, b, imu, jt.reshape(*imu.shape[:2], 72), ge, b0, film, dev).reshape(*imu.shape[:2], 24, 3)
            n = int(m.sum())
            add(s, "pos_cm_all", float(((p - jt).norm(dim=-1).mean(-1) * m).sum()) * 100, n)
            # best per-sequence-window global scale of the prediction (scale-invariant remainder)
            num = (p * jt).sum((-1, -2)); den = (p * p).sum((-1, -2)).clamp_min(1e-9)
            sc = (num.sum(1) / ((p * p).sum((-1, -2)) * m).sum(1)) if False else None
            w = ((p * jt).sum((1, 2, 3)) / (p * p).sum((1, 2, 3)).clamp_min(1e-9))[:, None, None, None]
            add(s, "pos_cm_after_best_scale", float((((w * p) - jt).norm(dim=-1).mean(-1) * m).sum()) * 100, n)
            for name, pairs in BONES.items():
                for (u, v) in pairs:
                    bp, bt = p[:, :, v] - p[:, :, u], jt[:, :, v] - jt[:, :, u]
                    cos = (bp * bt).sum(-1) / (bp.norm(dim=-1) * bt.norm(dim=-1)).clamp_min(1e-9)
                    add(s, f"dir_deg:{name}", float((torch.rad2deg(torch.acos(cos.clamp(-1, 1))) * m).sum()), n)
                    add(s, f"len_rel:{name}", float((((bp.norm(dim=-1) - bt.norm(dim=-1)).abs() / bt.norm(dim=-1).clamp_min(1e-9)) * m).sum()), n)
res = {s: {k: v[0] / v[1] for k, v in d.items()} for s, d in acc.items()}
# average the 2 sides (each bone name was added twice)
print(json.dumps(res, indent=1))
Path("results/line14/C6").mkdir(parents=True, exist_ok=True); json.dump(res, open(f"results/line14/C6/dirdiag_s{a.seed}.json", "w"), indent=1)
