"""Ceiling of any geometry input for joint positions (adult validation identities, eval-only, frozen Joints of one seed).
For each Joints variant, root-relative 24-joint position error (cm) is split by rebuilding the skeleton from bone directions and lengths:
  as_pred      the prediction itself
  true_len     predicted bone DIRECTIONS + TRUE bone lengths      (what perfect geometry could achieve)
  true_dir     TRUE bone directions + predicted bone lengths      (what perfect posture would leave)
  dir_deg      mean bone-direction error over the 23 bones (deg)
usage: python scripts_run/line14_C11_ceiling.py --seed 42 --gpu 0
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
import mobileposer.articulate as art
from mobileposer.config import paths
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line9.models import Line9Joints, GeometryStats
from mobileposer.line14.geom import GeomProvider
from mobileposer.line14.models import FilmJoints
import line14_poser_chain as C

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}")
EST = "data_local/line14_est_r2_b90.pt"
VARIANTS = {"B0": (None, "checkpoints/line14/E5/r1_b0_s{s}.pt", None), "PivotPrint estp6": ("estp6", "checkpoints/line14/E18/estp6_oracle_film_s{s}.pt", EST),
            "height": ("height", "checkpoints/line14/C9a/height_oracle_film_s{s}.pt", None), "height+PivotPrint concat": ("hestp6", "checkpoints/line14/C9a/hestp6_oracle_film_s{s}.pt", EST),
            "height+PivotPrint residual": ("hrestp6", "checkpoints/line14/C9ar/hrestp6_oracle_film_s{s}.pt", EST)}
parent = art.model.ParametricModel(paths.smpl_file, device="cpu").parent
rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("validation",))
store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
val = PairedWindowDataset(rec["validation"], 125, store=store, file_cache_size=2, sequence_cache_size=64)
vd, _ = C.loader(val, 128, a.seed, False)

def rebuild(root, dirs, lens):
    out = [root]
    for j in range(1, 24):
        out.append(out[parent[j]] + dirs[:, :, j] * lens[:, :, j:j + 1])
    return torch.stack(out, dim=2)

def load(kind, path, est):
    ck = torch.load(path.format(s=a.seed), map_location="cpu", weights_only=False)
    st = GeometryStats(ck["geometry_mean"], ck["geometry_std"])
    if kind is None:
        m = Line9Joints("b0", geom_dim=len(ck["geometry_mean"]), geometry_stats=st)
        prov = None
    else:
        m = FilmJoints(len(ck["geometry_mean"]), st); prov = GeomProvider(kind, 3.0, est, None)
    m.load_state_dict(ck["model"]); return m.to(dev).eval(), prov

res = {}
for name, (kind, path, est) in VARIANTS.items():
    m, prov = load(kind, path, est); acc = {}
    with torch.no_grad():
        for b in vd:
            imu = b["target_imu"].to(dev); mask = b["target_mask"].to(dev); L = b["target_lengths"].tolist()
            t = b["target_joints"][..., :3].to(dev)                         # [B,T,24,3]
            with torch.autocast(device_type="cuda"):
                p = (m(imu, L) if prov is None else m(imu, L, prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=a.seed).to(dev))).float()
            p = p.reshape(*t.shape)
            def bones(x):
                v = torch.zeros_like(x); l = torch.zeros(*x.shape[:3], device=dev)
                for j in range(1, 24):
                    d = x[:, :, j] - x[:, :, parent[j]]; l[:, :, j] = d.norm(dim=-1); v[:, :, j] = d / l[:, :, j:j + 1].clamp_min(1e-9)
                return v, l
            pd, pl = bones(p); td, tl = bones(t)
            variants = {"as_pred": p, "true_len": rebuild(p[:, :, 0], pd, tl), "true_dir": rebuild(t[:, :, 0], td, pl)}
            n = int(mask.sum())
            for k, x in variants.items():
                acc.setdefault(k, [0.0, 0]); acc[k][0] += float(((x - t).norm(dim=-1).mean(-1) * mask).sum()) * 100; acc[k][1] += n
            cos = (pd[:, :, 1:] * td[:, :, 1:]).sum(-1).clamp(-1, 1)
            acc.setdefault("dir_deg", [0.0, 0]); acc["dir_deg"][0] += float((torch.rad2deg(torch.acos(cos)).mean(-1) * mask).sum()); acc["dir_deg"][1] += n
    res[name] = {k: v[0] / v[1] for k, v in acc.items()}
    print(name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
Path("results/line14/C11").mkdir(parents=True, exist_ok=True); json.dump(res, open(f"results/line14/C11/ceiling_s{a.seed}.json", "w"), indent=1)
