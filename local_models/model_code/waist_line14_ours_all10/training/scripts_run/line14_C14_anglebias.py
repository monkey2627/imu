"""Angle-level headroom of a PERSON-LEVEL correction: how much of the end-to-end Poser's global-orientation error is a constant per-identity offset?
Eval-only, adult validation identities, one seed.  Error rotation e = log(R_pred R_gt^T) (world) -> pelvis-sensor frame; per identity the mean error vector of each of the
16 reduced joints is estimated on half of the identity's SEQUENCES and removed from the other half (and vice versa).  Reported: RMS angle (deg) before/after.
usage: python scripts_run/line14_C14_anglebias.py --run C5/e2e_film --seed 42 --gpu 0"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
import mobileposer.articulate as art
from mobileposer.articulate import math as AM
from mobileposer.config import joint_set
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line9.models import Line9Joints, GeometryStats
from mobileposer.line14.geom import GeomProvider
from mobileposer.line14.models import FilmJoints
import line14_poser_chain as C

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="C5/e2e_film"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--geom", default="estp6"); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}"); EST = "data_local/line14_est_r2_b90.pt"
st = torch.load(f"checkpoints/line14/{a.run}_s{a.seed}.pt", map_location="cpu", weights_only=False)
film = a.run.endswith("film")
if film:
    prov = GeomProvider(a.geom, 3.0, EST, None)
    jn = C.load_frozen(a.seed, dev)[1] if a.geom == "estp6" else None
else:
    prov = None; jn = C.load_frozen(a.seed, dev)[0]
jn.load_state_dict(st["jnet"]); jn.eval()
poser = C.ChainPoser(132).to(dev); poser.load_state_dict(st["poser"]); poser.eval()
rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("validation",))
store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
val = PairedWindowDataset(rec["validation"], 125, store=store, file_cache_size=2, sequence_cache_size=64)
vd, _ = C.loader(val, 128, a.seed, False)
R = joint_set.reduced; parent = art.model.ParametricModel(__import__("mobileposer.config", fromlist=["paths"]).paths.smpl_file, device="cpu").parent
E = defaultdict(list)
with torch.no_grad():
    for b in vd:
        imu = b["target_imu"].to(dev); mask = b["target_mask"].to(dev); L = b["target_lengths"].tolist()
        ge = prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=a.seed).to(dev) if prov else None
        with torch.autocast(device_type="cuda"):
            j = jn(imu, L, ge) if prov else jn(imu, L)
            p = poser(torch.cat((j.to(imu.dtype), imu), -1), L)
        p = p.float().reshape(*p.shape[:2], 16, 6)
        Rp = AM.r6d_to_rotation_matrix(p.reshape(-1, 6)).reshape(*p.shape[:2], 16, 3, 3)
        y = C.global_targets(b["target_joints"][..., 3:].to(dev), parent).reshape(*p.shape[:2], 16, 6)
        Rg = AM.r6d_to_rotation_matrix(y.reshape(-1, 6)).reshape(*p.shape[:2], 16, 3, 3)
        Re = Rp @ Rg.transpose(-1, -2)
        ev = AM.rotation_matrix_to_axis_angle(Re.reshape(-1, 3, 3)).reshape(*p.shape[:2], 16, 3)
        Rpel = imu[..., 51:60].reshape(*imu.shape[:2], 3, 3)
        ep = (Rpel.transpose(-1, -2).unsqueeze(2) @ ev.unsqueeze(-1)).squeeze(-1) * mask[:, :, None, None]
        for k in range(len(ep)):
            n = int(mask[k].sum()); e = ep[k, :n]
            E[b["identity"][k]].append((e.sum(0).cpu(), n, (e ** 2).sum((0, 2)).cpu(), b["target_sequence"][k]))
raw = after = n_tot = 0.0
for ident, ws in E.items():
    seqs = sorted({w[3] for w in ws})
    if len(seqs) < 4: continue
    sa = set(seqs[0::2]); A, B = [w for w in ws if w[3] in sa], [w for w in ws if w[3] not in sa]
    for fit, ev_ in ((A, B), (B, A)):
        nf = sum(w[1] for w in fit); bias = sum(w[0] for w in fit) / nf
        ne = sum(w[1] for w in ev_); me = sum(w[0] for w in ev_) / ne; sq = sum(w[2] for w in ev_) / ne
        sqa = sq - 2 * (bias * me).sum(-1) + (bias ** 2).sum(-1)
        raw += float(sq.mean()) * ne; after += float(sqa.mean()) * ne; n_tot += ne
r = {"run": a.run, "seed": a.seed, "rms_deg_raw": float(np.degrees((raw / n_tot) ** 0.5)), "rms_deg_after_person_bias": float(np.degrees((after / n_tot) ** 0.5)), "mse_reduction_pct": 100 * (1 - after / raw)}
print(json.dumps(r)); Path("results/line14/C14").mkdir(parents=True, exist_ok=True); json.dump(r, open(f"results/line14/C14/anglebias_{a.run.replace('/', '_')}_s{a.seed}.json", "w"))
