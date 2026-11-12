"""Diagnostics for the chain experiment (eval-only, validation identities, uses the C1/C2 `joints` Posers):
 (1) Ang / SIP vs i.i.d. joint noise sigma (GT + noise) -> sensitivity curve of the Poser to joint-position error;
 (2) joint-position error (cm, root-relative) of the frozen B0 and FiLM(estp6) Joints on exactly these windows, per joint group;
 (3) the same Poser evaluated on B0 / FiLM predictions, for mapping 'real Joints error' to an equivalent i.i.d. sigma.
usage: python scripts_run/line14_C3_diag.py --tag C1 --seed 42 --gpu 0
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mobileposer.articulate as art
from mobileposer.config import paths
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line14.geom import GeomProvider
import line14_poser_chain as C

GROUPS = {"hip": [1, 2], "knee": [4, 5], "shoulder": [16, 17], "elbow": [18, 19], "sensed_wrist_ankle": [20, 21, 7, 8]}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="C1"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
kw = dict(max_identities_per_fold=3, max_sequences_per_identity=3) if a.smoke else {}
rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("validation",), **kw)
store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
val = PairedWindowDataset(rec["validation"], 125, store=store, file_cache_size=2, sequence_cache_size=64, max_windows=16 if a.smoke else None)
vd, _ = C.loader(val, 128, a.seed, False)
body = art.model.ParametricModel(paths.smpl_file, device=device)
from mobileposer.evaluate_standard import StdEvaluator
ev = StdEvaluator(device)
prov = GeomProvider("estp6", 3.0, "data_local/line14_est_r2_b90.pt")
b0, film = C.load_frozen(a.seed, device)
model = C.ChainPoser(132).to(device)
model.load_state_dict(torch.load(f"checkpoints/line14/{a.tag}/joints_s{a.seed}.pt", map_location=device, weights_only=False)["model"])
args = SimpleNamespace(variant="joints", seed=a.seed)

out = {"curve": {}, "joint_err_cm": {}}
for s in ["gt", "gtn1", "gtn2", "gtn3", "gtn4", "gtn5", "gtn6", "gtn8", "b0", "film"]:
    m = C.evaluate(model, vd, args, device, body, ev, prov, None, s, b0, film, a.seed)
    out["curve"][s] = {k: m[k] for k in ("sip", "ang", "pos", "mesh")}
    print(s, {k: round(v, 3) for k, v in out["curve"][s].items()}, flush=True)

# joint-position error of the Joints themselves
acc = {s: {g: [0.0, 0] for g in list(GROUPS) + ["all"]} for s in ("b0", "film")}
with torch.no_grad():
    for b in vd:
        imu = b["target_imu"].to(device); mask = b["target_mask"].to(device)
        jt = b["target_joints"][..., :3].to(device)
        ge = prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=a.seed).to(device)
        for s in acc:
            p = C.joint_source(s, b, imu, jt.reshape(*imu.shape[:2], 72), ge, b0, film, device).reshape(*imu.shape[:2], 24, 3)
            e = (p - jt).norm(dim=-1) * 100
            for g, idx in list(GROUPS.items()) + [("all", list(range(24)))]:
                acc[s][g][0] += float((e[:, :, idx].mean(-1) * mask).sum()); acc[s][g][1] += int(mask.sum())
out["joint_err_cm"] = {s: {g: v[0] / v[1] for g, v in acc[s].items()} for s in acc}
print(json.dumps(out["joint_err_cm"], indent=1))
Path(f"results/line14/{a.tag}").mkdir(parents=True, exist_ok=True)
json.dump(out, open(f"results/line14/{a.tag}/diag_s{a.seed}{'_smoke' if a.smoke else ''}.json", "w"), indent=1)
