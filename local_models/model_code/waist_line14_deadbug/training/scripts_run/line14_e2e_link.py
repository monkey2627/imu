"""Line 14 (C19): the SENSOR-TO-BONE CONNECTION recovered by calibration, and its effect on the rotation estimate.

The connection has two parts that calibration actions can reveal:
  (a) the sensor-to-bone ORIENTATION offset R_off (3 DoF per sensor)  -- the measured orientation is  R_meas = R_bone @ R_off  (line-9 I2S encoder, ~2.5 deg error)
  (b) the BODY geometry seen by the sensors (sensor -> joint vectors, scale)  -- PivotPrint estp6 (line-14 estimator, FiLM-conditioned Joints)
Data: line9_store_calib6 (AMASS identities with a fixed per-identity random sensor offset: orientation RMS 5 deg, lever arm 3 cm, pelvis yaw gauge fixed);
both estimators were trained on calibration sessions of the training identities only and predict every identity (validation identities unseen).
Everything is trained FROM SCRATCH end-to-end (Joints + Poser jointly: pose loss + joint loss), identical budget in every arm:

  --ori none|est|true   none: raw measured orientations;  est: R_meas @ R_est^T (calibration estimate);  true: R_meas @ R_off^T (oracle)
  --geom none|estp6     Joints conditioned on the estimated PivotPrint (FiLM) or not
arms:  none/none (B0)  | none/estp6 (geometry only) | est/none (I2S only) | est/estp6 (full connection) | true/none, true/estp6 (oracles)
usage: LINE9_GAUGE=1 python scripts_run/line14_e2e_link.py --ori est --geom estp6 --seed 42 --gpu 0 --tag C19
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import mobileposer.articulate as art
from mobileposer.config import paths
from mobileposer.line9.calib import identity_offsets
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line9.models import GeometryStats, Line9Joints, fit_geometry_stats
from mobileposer.line14.geom import GeomProvider
from mobileposer.line14.models import FilmJoints
from types import SimpleNamespace
import line14_poser_chain as C

PRED = "/home/duanyuhan/dyh/motion/SparseIMU/results/line9/enc_v2/pred_phys_h128_big_s{seed}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ori", choices=["none", "est", "true"], required=True); ap.add_argument("--geom", choices=["none", "estp6", "height", "hestp6", "hrestp6"], required=True)
    ap.add_argument("--ori-file", default=None, help="torch file with {\"R\": identity -> [5,3,3]} (default: line-9 I2S encoder predictions)"); ap.add_argument("--est-file", default="data_local/line14_est_r2_b90.pt"); ap.add_argument("--scale-range", default=""); ap.add_argument("--name-suffix", default="")
    ap.add_argument("--lever", choices=["none", "mean", "est", "true"], default="none", help="PivotPrint as physics: lever-arm compensated joint accelerations (36 extra input columns); mean = population-mean vectors, est = estimated, true = ground truth")
    ap.add_argument("--lever-keep", choices=["all", "arms", "legs"], default="all", help="which limb sensors get the lever-arm compensation (shank compensation hurt on real data in the baseline project)")
    ap.add_argument("--ori-noise-deg", type=float, default=0.0, help="ori true: add a fixed per-identity random rotation to R_off so that the MEAN angle error equals this value (estimator-like profile: limbs ~1.2x, pelvis ~0.3x) -> precision/accuracy curve")
    ap.add_argument("--lever-file", default=None, help="est: the assembled geometry estimate file (identity -> {mu [S,24] m})")
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--tag", default="C19")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01); ap.add_argument("--joint-w", type=float, default=1.0); ap.add_argument("--window", type=int, default=125)
    ap.add_argument("--imu-store", default="data/line9_store_calib6"); ap.add_argument("--rot-deg", type=float, default=5.0); ap.add_argument("--lever-cm", type=float, default=3.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("LINE9_GAUGE") == "1", "set LINE9_GAUGE=1 (pelvis yaw gauge)"
    if a.smoke:
        a.epochs = 1
    if a.scale_range:
        C.CFG["sr"] = tuple(float(x) for x in a.scale_range.split(","))
    C.seed_all(a.seed); device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    kw = dict(max_identities_per_fold=3, max_sequences_per_identity=3) if a.smoke else {}
    rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("train", "validation"), **kw)
    store = C.ChainStore(Path(a.imu_store), Path("data/line9_store_pose6d"))
    mk = lambda r: PairedWindowDataset(r, a.window, store=store, file_cache_size=2, sequence_cache_size=64, max_windows=16 if a.smoke else None)
    train, val = mk(rec["train"]), mk(rec["validation"])
    body = art.model.ParametricModel(paths.smpl_file, device=device); parent = body.parent
    from mobileposer.evaluate_standard import StdEvaluator
    ev = StdEvaluator(device)
    ids = sorted({r.identity for r in rec["train"] + rec["validation"]})
    if a.ori == "est":
        Rmap = torch.load(a.ori_file or PRED.format(seed=a.seed), weights_only=False)["R"]
    elif a.ori == "true":
        Rmap = {i: identity_offsets(i, a.rot_deg, a.lever_cm)[0] for i in ids}
        if a.ori_noise_deg > 0:
            import hashlib
            prof = torch.tensor([1.16, 1.27, 1.16, 1.10, 0.32])
            for i in ids:
                hh = hashlib.sha256(f"{i}|orinoise|{a.ori_noise_deg}".encode()).digest()
                gg = torch.Generator().manual_seed(int.from_bytes(hh[:8], "little") % (2 ** 63))
                v = torch.randn(5, 3, generator=gg) * (np.deg2rad(a.ori_noise_deg) / 1.596) * prof[:, None]
                Rmap[i] = Rmap[i] @ art.math.axis_angle_to_rotation_matrix(v).reshape(5, 3, 3)
    else:
        Rmap = None

    rmap = None
    if a.lever != "none":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from line14_physfeat import lever_features, prox_vectors
        r2_true = GeomProvider("r2", 3.0, None, C.CFG["sr"] if a.lever == "true" else None)
        r2_unscaled = GeomProvider("r2", 3.0, None, None)
        betas = {}
        for ds in (train, val):
            for rr in (rec["train"] if ds is train else rec["validation"]):
                betas.setdefault(rr.identity, ds._get_geometry(rr)[0])
        if a.lever == "true":
            rmap = {i: prox_vectors(r2_true.base(bt, i)) for i, bt in betas.items()}
        elif a.lever == "mean":
            tr_ids = sorted({rr.identity for rr in rec["train"]})
            m = torch.stack([prox_vectors(r2_unscaled.base(betas[i], i)) for i in tr_ids]).mean(0)
            rmap = {i: m for i in betas}
        else:
            E = torch.load(a.lever_file, weights_only=False)
            rmap = {i: prox_vectors(E[i]["mu"].mean(0).float()) for i in betas}
    EXTRA = 36 if a.lever != "none" else 0
    KEEP = {"all": (1, 1, 1, 1), "arms": (1, 1, 0, 0), "legs": (0, 0, 1, 1)}[a.lever_keep]

    def hook(imu, b):
        if Rmap is not None:
            R = torch.stack([Rmap[i] for i in b["identity"]]).to(imu.device).float()                       # [B,5,3,3]
            ori = imu[..., 15:60].reshape(*imu.shape[:2], 5, 3, 3) @ R[:, None].transpose(-1, -2)
            imu = torch.cat((imu[..., :15], ori.reshape(*imu.shape[:2], 45)), dim=-1)
        if rmap is not None:
            r = torch.stack([rmap[i] for i in b["identity"]])
            imu = torch.cat((imu, lever_features(imu, r, b["target_lengths"].tolist(), keep=KEEP)), dim=-1)
        return imu
    C.CFG["hook"] = hook

    kind = a.geom if a.geom != "none" else "estp6"
    prov = GeomProvider(kind, 3.0, a.est_file if "estp" in kind else None, None if "estp" in kind else C.CFG["sr"])
    if a.geom != "none":
        rows, seen = [], set()
        for r in rec["train"]:
            if r.identity not in seen:
                beta, _ = train._get_geometry(r); rows.append(prov.base(beta, r.identity)); seen.add(r.identity)
        jnet = FilmJoints(prov.dim, fit_geometry_stats(torch.stack(rows))).to(device); src = "film"
    else:
        jnet = Line9Joints("b0", geom_dim=1, geometry_stats=GeometryStats(torch.zeros(1), torch.ones(1))).to(device); src = "b0"
    if EXTRA:
        jnet.linear1 = nn.Linear(60 + EXTRA, 256).to(device); jnet.input_dim = 60 + EXTRA
    poser = C.ChainPoser(132 + EXTRA).to(device)
    td, ts = C.loader(train, a.batch, a.seed, True); vd, _ = C.loader(val, a.batch, a.seed, False)
    opt = torch.optim.AdamW(list(jnet.parameters()) + list(poser.parameters()), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    name = f"ori-{a.ori}_geom-{a.geom}_s{a.seed}" + (f"_{a.name_suffix}" if a.name_suffix else "")
    out = Path("results/line14") / a.tag / f"{name}.json"; ck = Path("checkpoints/line14") / a.tag / f"{name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True); ck.parent.mkdir(parents=True, exist_ok=True)

    def fwd(b, fixed_seed):
        imu, jt = C.get_inputs(b, device); L = b["target_lengths"].tolist()
        if a.geom != "none":
            g = prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=fixed_seed).to(device); j = jnet(imu, L, g)
        else:
            j = jnet(imu, L)
        return jt, j, poser(torch.cat((j.to(imu.dtype), imu), -1), L)

    def val_score():
        poser.eval(); jnet.eval(); ve = vn = 0; je = jn = 0
        with torch.no_grad():
            for b in vd:
                mask = b["target_mask"].to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    jt, j, p = fwd(b, a.seed)
                y = C.global_targets(b["target_joints"][..., 3:].to(device), parent)
                ve += float(((p.float() - y).square().mean(-1) * mask).sum()); vn += int(mask.sum())
                je += float(((j.float() - jt).reshape(*mask.shape, 24, 3).norm(dim=-1).mean(-1) * mask).sum()) * 100; jn += int(mask.sum())
        return ve / vn, je / jn

    best, hist, t0 = float("inf"), [], time.time()
    for epoch in range(a.epochs):
        ts.set_epoch(epoch); poser.train(); jnet.train(); tot = n = 0
        for b in td:
            mask = b["target_mask"].to(device)
            with torch.no_grad():
                y = C.global_targets(b["target_joints"][..., 3:].to(device), parent)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                jt, j, p = fwd(b, None)
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
    metrics = C.evaluate(poser, vd, args, device, body, ev, prov, None, src, jnet if src == "b0" else None, jnet if src == "film" else None, a.seed)
    _, je = val_score()
    res = {"status": "smoke_only" if a.smoke else "formal_validation_only", "ori": a.ori, "geom": a.geom, "seed": a.seed,
           "protocol": {"epochs": a.epochs, "batch": a.batch, "lr": a.lr, "imu_store": a.imu_store, "window": a.window, "pred": PRED.format(seed=a.seed) if a.ori == "est" else None,
                        "target_window_digest_validation": val.target_window_digest(), "test_accessed": False},
           "best_epoch": st["epoch"], "joint_err_cm_final": je, "metrics": {src: metrics}, "history": hist, "elapsed_seconds": time.time() - t0}
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)
    print(json.dumps({"run": name, **{m: round(metrics[m], 3) for m in ("sip", "ang", "pos", "mesh")}, "joint_err_cm": round(je, 3)}))


if __name__ == "__main__":
    main()
