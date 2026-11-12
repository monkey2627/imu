"""Line 14 end-to-end chain (plan C, joint training): geometry -> FiLM Joints -> Poser, trained together with the pose loss
(+ lambda * joint-position loss), validation-only, synthetic.

Two arms that differ ONLY in the Joints network:
  --joints-net b0     Joints without geometry (E5 r1_b0), trainable
  --joints-net film   FilmJoints with the estimated PivotPrint estp6 (E18), trainable, geometry = random calibration session per step
Both start from the pretrained Joints (same seed) and from the C1 `joints` Poser (trained on GT joints + 4 cm noise), then are fine-tuned
jointly: x = cat(Joints(imu, geom), imu) -> Poser -> loss = pose_mse(+temporal) + lambda * joint_mse(Joints output vs individual-shape joints).
Evaluation: SIP / Ang / Pos / Mesh of the standard evaluator on the validation identities (the arm's own Joints output feeds the Poser).
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
import mobileposer.articulate as art
from mobileposer.config import paths
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line14.geom import GeomProvider
from types import SimpleNamespace
import line14_poser_chain as C

ap = argparse.ArgumentParser()
ap.add_argument("--joints-net", choices=["b0", "film"], required=True)
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--tag", default="C5")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--joint-w", type=float, default=1.0, help="weight of the joint-position loss (0 = pose loss only)")
ap.add_argument("--init-poser-tag", default="C1", help="checkpoints/line14/<tag>/joints_s<seed>.pt, or 'none' for a fresh Poser")
ap.add_argument("--window", type=int, default=125)
ap.add_argument("--film-ckpt", default=None); ap.add_argument("--film-geom", default="estp6"); ap.add_argument("--scale-range", default=""); ap.add_argument("--est-file", default="data_local/line14_est_r2_b90.pt")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.epochs = 1
if a.scale_range:
    C.CFG["sr"] = tuple(float(x) for x in a.scale_range.split(",")); C.CFG["b0"] = "checkpoints/line14/E19/r1_b0_s{seed}.pt"; C.CFG["film"] = f"checkpoints/line14/E19/{a.film_geom}_oracle_film_s{{seed}}.pt"
if a.film_ckpt:
    C.CFG["film"] = a.film_ckpt
C.seed_all(a.seed); device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
kw = dict(max_identities_per_fold=3, max_sequences_per_identity=3) if a.smoke else {}
rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("train", "validation"), **kw)
store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
mk = lambda r: PairedWindowDataset(r, a.window, store=store, file_cache_size=2, sequence_cache_size=64, max_windows=16 if a.smoke else None)
train, val = mk(rec["train"]), mk(rec["validation"])
body = art.model.ParametricModel(paths.smpl_file, device=device); parent = body.parent
from mobileposer.evaluate_standard import StdEvaluator
ev = StdEvaluator(device)
prov = GeomProvider(a.film_geom, 3.0, a.est_file if "estp" in a.film_geom else None, None if "estp" in a.film_geom else C.CFG["sr"])
b0, film = C.load_frozen(a.seed, device)
jnet = film if a.joints_net == "film" else b0
poser = C.ChainPoser(132).to(device)
if a.init_poser_tag != "none":
    poser.load_state_dict(torch.load(f"checkpoints/line14/{a.init_poser_tag}/joints_s{a.seed}.pt", map_location=device, weights_only=False)["model"])
td, ts = C.loader(train, a.batch, a.seed, True); vd, _ = C.loader(val, a.batch, a.seed, False)
params = list(poser.parameters()) + list(jnet.parameters())
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
name = f"e2e_{a.joints_net}" + (f"_{a.film_geom}" if a.film_geom != "estp6" and a.joints_net == "film" else "") + (f"_w{a.joint_w:g}" if a.joint_w != 1.0 else "") + f"_s{a.seed}"
out = Path("results/line14") / a.tag / f"{name}.json"; ck = Path("checkpoints/line14") / a.tag / f"{name}.pt"
out.parent.mkdir(parents=True, exist_ok=True); ck.parent.mkdir(parents=True, exist_ok=True)


def forward(b, train_mode):
    imu, _ = C.get_inputs(b, device); L = b["target_lengths"].tolist()
    ge = prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=None if train_mode else a.seed).to(device) if a.joints_net == "film" else None
    j = jnet(imu, L, ge) if a.joints_net == "film" else jnet(imu, L)
    p = poser(torch.cat((j.to(imu.dtype), imu), -1), L)
    return j, p


def val_score():
    poser.eval(); jnet.eval(); ve = vn = 0; je = jn = 0
    with torch.no_grad():
        for b in vd:
            mask = b["target_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                j, p = forward(b, False)
            y = C.global_targets(b["target_joints"][..., 3:].to(device), parent)
            _, jt = C.get_inputs(b, device)
            ve += float(((p.float() - y).square().mean(-1) * mask).sum()); vn += int(mask.sum())
            je += float(((j.float() - jt).reshape(*mask.shape, 24, 3).norm(dim=-1).mean(-1) * mask).sum()) * 100; jn += int(mask.sum())
    return ve / vn, je / jn


best, hist, t0 = float("inf"), [], time.time()
s0, j0 = val_score(); print(json.dumps({"epoch": 0, "val_mse_6d": s0, "joint_err_cm": j0}), flush=True)
for epoch in range(a.epochs):
    ts.set_epoch(epoch); poser.train(); jnet.train(); tot = n = 0
    for b in td:
        mask = b["target_mask"].to(device)
        with torch.no_grad():
            y = C.global_targets(b["target_joints"][..., 3:].to(device), parent)
        _, jt = C.get_inputs(b, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            j, p = forward(b, True)
            mse = ((p.float() - y).square() * mask[:, :, None]).sum() / (mask.sum() * 96)
            acc = p[:, 2:] + p[:, :-2] - 2 * p[:, 1:-1]; tm = mask[:, 2:] & mask[:, :-2] & mask[:, 1:-1]
            loss = mse + 1e-5 * (acc.abs() * tm[:, :, None]).sum() / tm.sum().clamp_min(1)
            if a.joint_w > 0:
                loss = loss + a.joint_w * ((j.float() - jt).square() * mask[:, :, None]).sum() / (mask.sum() * 72)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += float(loss) * int(mask.sum()); n += int(mask.sum())
    s, je = val_score()
    hist.append({"epoch": epoch + 1, "train_loss": tot / max(n, 1), "val_mse_6d": s, "joint_err_cm": je}); print(json.dumps({"run": name, **hist[-1]}), flush=True)
    if s < best:
        best = s; torch.save({"poser": poser.state_dict(), "jnet": jnet.state_dict(), "epoch": epoch + 1, "score": s}, ck, pickle_protocol=4)
st = torch.load(ck, map_location=device, weights_only=False); poser.load_state_dict(st["poser"]); jnet.load_state_dict(st["jnet"])
args = SimpleNamespace(variant="joints", seed=a.seed)
metrics = C.evaluate(poser, vd, args, device, body, ev, prov, None, a.joints_net, jnet if a.joints_net == "b0" else b0, jnet if a.joints_net == "film" else film, a.seed)
_, je = val_score()
res = {"status": "smoke_only" if a.smoke else "formal_validation_only", "arm": a.joints_net, "seed": a.seed, "joint_w": a.joint_w,
       "protocol": {"epochs": a.epochs, "batch": a.batch, "lr": a.lr, "init_poser_tag": a.init_poser_tag, "window": a.window,
                    "target_window_digest_validation": val.target_window_digest(), "test_accessed": False},
       "best_epoch": st["epoch"], "joint_err_cm_final": je, "joint_err_cm_before": j0, "val_mse_before": s0, "metrics": {a.joints_net: metrics}, "history": hist,
       "elapsed_seconds": time.time() - t0}
tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)
print(json.dumps({"run": name, **{m: round(metrics[m], 3) for m in ("sip", "ang", "pos", "mesh")}, "joint_err_cm": round(je, 3)}))
