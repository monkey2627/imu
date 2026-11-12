"""Line 14 (C20): the sensor-to-bone LINK estimator -- one network that reads a ~13 s calibration session and returns BOTH
  * the per-sensor orientation offset R_off (5 x SO(3); measured orientation = R_bone @ R_off, pelvis yaw gauge fixed), and
  * the body geometry seen by the sensors: R2 = 8 sensor->joint vectors (24-d, cm; PivotPrint is its low-dimensional PCA coordinates, scale included).

modes
  rot  rotation head only                                   (the line-9 I2S estimator, retrained on the same data)
  geo  geometry head only                                   (the line-14 PivotPrint estimator, retrained on the same data)
  uni  one trunk, both heads, joint loss                    (multi-task "link")
  cas  coarse-to-fine: stage 2 of a cascade.  The calibration session is first corrected with the stage-1 rotation estimate
       (training: true R_off perturbed by --cas-noise-deg; evaluation: stage-1 prediction), then the geometry is read from the corrected session.

Data = virtual subjects of line14_build_cal_set.py (dynamic 13 s protocol, per-subject installation offsets with --rot-deg RMS) and, for the downstream
identities, line14_build_cal_identities.py (offset_key = identity, i.e. the SAME offsets that are baked into data/line9_store_calib6).
A global body scale s in [lo,hi] can be applied (accelerations and geometry targets scale with s, orientations unchanged): --scale-range lo,hi.

usage: LINE9_GAUGE=1 python scripts_run/line14_link_estimator.py --mode uni --data data_local/line15_cal_virtual --seed 42 --gpu 0 --tag C20 \
         --rot-deg 5 --scale-range 0.5,1.2 --ids-scale 0.5,1.2 --predict data_local/line15_cal_ids/train.pt,data_local/line15_cal_ids/val.pt --pred-name uni_full
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mobileposer.line9.enc2 import phys_features
from mobileposer.line9.robust_calib import heading_normalize
from mobileposer.line9.calib import identity_offsets
from mobileposer.line14.geom import GeomProvider, scale_ident
from mobileposer.line14.link import LinkEncoder
import line14_estimate_r1 as E1


def geodesic(R1, R2):
    """mean rotation angle (rad) between R1,R2 [...,3,3]"""
    c = ((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


def rot_targets(blob, rot_deg, lever_cm, offset_key):
    keys = blob["key"] if offset_key == "key" else blob["identity"]
    return torch.stack([identity_offsets(k, rot_deg, lever_cm, gauge=True)[0] for k in keys])


def correct_imu(imu, R):
    """measured orientation @ R^T  (R = estimate of R_off) on the raw session imu [N,T,60]"""
    N, T = imu.shape[:2]
    ori = imu[..., 15:60].reshape(N, T, 5, 3, 3) @ R[:, None].transpose(-1, -2)
    return torch.cat((imu[..., :15], ori.reshape(N, T, 45)), dim=-1)


def random_rot(n, deg, gen):
    from mobileposer.articulate import math as M
    v = torch.randn(n, 5, 3, generator=gen) * (np.deg2rad(deg) / np.sqrt(3.0))
    return M.axis_angle_to_rotation_matrix(v.reshape(-1, 3)).reshape(n, 5, 3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rot", "geo", "uni", "cas"], required=True)
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="C20"); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rot-deg", type=float, default=5.0); ap.add_argument("--lever-cm", type=float, default=3.0); ap.add_argument("--offset-key", default="key", choices=["key", "identity"])
    ap.add_argument("--scale-range", default="1,1"); ap.add_argument("--ids-scale", default="")
    ap.add_argument("--w-rot", type=float, default=10.0, help="weight of the mean geodesic angle (rad) next to the geometry NLL (uni)")
    ap.add_argument("--stage1", default="", help="cas: checkpoint of the stage-1 rotation estimator"); ap.add_argument("--cas-noise-deg", type=float, default=2.5)
    ap.add_argument("--exclude-ids", default=""); ap.add_argument("--predict", default=""); ap.add_argument("--pred-name", default="")
    ap.add_argument("--pred-offset-key", default="identity", choices=["key", "identity"]); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    lo, hi = [float(x) for x in a.scale_range.split(",")]
    prov = GeomProvider("r1")
    tr = torch.load(a.data / "train.pt", weights_only=False); va = torch.load(a.data / "val.pt", weights_only=False)
    if a.smoke:
        for d in (tr, va):
            for k in ("imu", "phase_id", "beta", "lever", "psi", "keep", "amp", "dev"):
                if k in d: d[k] = d[k][:16]
            for k in ("key", "identity"): d[k] = d[k][:16]
        a.epochs = 1
    if a.exclude_ids:
        excl = set(json.load(open(a.exclude_ids)))
        keep = torch.tensor([i not in excl for i in tr["identity"]])
        for k in ("imu", "phase_id", "beta", "lever", "psi", "keep", "amp", "dev"):
            if k in tr: tr[k] = tr[k][keep]
        for k in ("key", "identity"): tr[k] = [x for x, kp in zip(tr[k], keep.tolist()) if kp]
    with_rot, with_geo = a.mode in ("rot", "uni"), a.mode in ("geo", "uni", "cas")
    if a.mode == "uni": with_rot = with_geo = True
    # ---- targets
    ytr, yva = E1.targets(tr, prov, "r2"), E1.targets(va, prov, "r2")
    Rtr, Rva = rot_targets(tr, a.rot_deg, a.lever_cm, a.offset_key), rot_targets(va, a.rot_deg, a.lever_cm, a.offset_key)
    s_tr = E1.scale_for(tr["key"], lo, hi, a.seed) if hi > lo else torch.ones(len(ytr)); s_va = E1.scale_for(va["key"], lo, hi, 7) if hi > lo else torch.ones(len(yva))
    imu_tr, ytr = E1.apply_scale(tr["imu"].float(), ytr, s_tr); imu_va, yva = E1.apply_scale(va["imu"].float(), yva, s_va)
    mtr, mva = E1.valid_mask(tr), E1.valid_mask(va)

    def feats(imu, fm=None, fs=None, mk=None):
        x = E1.make_inputs({"imu": imu}, "phys", a.stride)
        if fm is None:
            fm = (x * mk[..., None]).sum((0, 1)) / mk.sum().clamp_min(1)
            fs = (((x - fm) ** 2 * mk[..., None]).sum((0, 1)) / mk.sum().clamp_min(1)).sqrt().clamp_min(1e-4)
        return (x - fm) / fs * mk[..., None], fm, fs

    stage1 = None
    if a.mode == "cas":
        st1 = torch.load(a.stage1, map_location="cpu", weights_only=False)
        stage1 = LinkEncoder(135, with_rot=True, with_geo=False).to(dev); stage1.load_state_dict(st1["model"]); stage1.eval()

        def stage1_R(imu, ph, mk):
            x = ((E1.make_inputs({"imu": imu}, "phys", a.stride) - st1["fm"]) / st1["fs"] * mk[..., None])
            out = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    out.append(stage1(x[i:i + 64].to(dev), ph[i:i + 64].to(dev), mk[i:i + 64].to(dev))[1].float().cpu())
            return torch.cat(out)
        gen = torch.Generator().manual_seed(a.seed)
        imu_tr_in = correct_imu(imu_tr, Rtr @ random_rot(len(Rtr), a.cas_noise_deg, gen))           # training: true offset + noise
        imu_va_in = correct_imu(imu_va, stage1_R(imu_va, va["phase_id"].long(), mva))                # evaluation: stage-1 estimate
    else:
        imu_tr_in, imu_va_in = imu_tr, imu_va
    xtr, fm, fs = feats(imu_tr_in, mk=mtr); xva, _, _ = feats(imu_va_in, fm, fs, mva)
    ym, ys = ytr.mean(0), ytr.std(0).clamp_min(1e-3); ytr_n = (ytr - ym) / ys
    ph_tr, ph_va = tr["phase_id"].long(), va["phase_id"].long()
    model = LinkEncoder(xtr.shape[-1], geo_dim=ytr.shape[1], with_rot=with_rot, with_geo=with_geo, hidden=a.hidden, layers=a.layers).to(dev)
    D = ytr.shape[1]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(xtr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1))
    dl = DataLoader(TensorDataset(xtr, ph_tr, mtr, ytr_n, Rtr), batch_size=a.batch, shuffle=True, drop_last=False)
    out_dir = Path("results/line14") / a.tag; out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir = Path("checkpoints/line14") / a.tag; ck_dir.mkdir(parents=True, exist_ok=True)
    name = f"{a.mode}_s{a.seed}" + (f"_{a.pred_name}" if a.pred_name else "")
    eye = torch.eye(3).expand_as(Rva)
    rot_base = float(torch.rad2deg(geodesic(eye, Rva)).mean())
    geo_base = E1.metrics(ym[None].expand_as(yva), yva, torch.zeros_like(yva) + 3.0, ys)["vec_err_cm_mean"]

    def predict(x, ph, mk):
        model.eval(); mus, lvs, Rs = [], [], []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                g, R = model(x[i:i + 64].to(dev), ph[i:i + 64].to(dev), mk[i:i + 64].to(dev))
                if g is not None:
                    g = g.float().cpu(); mus.append(g[:, :D]); lvs.append(g[:, D:].clamp(-6, 4))
                if R is not None:
                    Rs.append(R.float().cpu())
        mu = torch.cat(mus) * ys + ym if mus else None
        lv = torch.cat(lvs) + 2 * torch.log(ys) if lvs else None
        return mu, lv, (torch.cat(Rs) if Rs else None)

    def evaluate():
        mu, lv, R = predict(xva, ph_va, mva); r = {}
        if mu is not None:
            r["geo"] = E1.metrics(mu, yva, lv, ys)
        if R is not None:
            r["rot_deg"] = float(torch.rad2deg(geodesic(R, Rva)).mean()); r["rot_deg_per_sensor"] = torch.rad2deg(geodesic(R, Rva)).mean(0).tolist()
        r["score"] = (r.get("rot_deg", 0.0) / rot_base if with_rot else 0.0) + (r["geo"]["vec_err_cm_mean"] / geo_base if with_geo else 0.0)
        return r

    best, hist, t0 = float("inf"), [], time.time()
    for epoch in range(a.epochs):
        model.train(); tot = n = 0
        for x, ph, mk, y, Rt in dl:
            x, ph, mk, y, Rt = x.to(dev), ph.to(dev), mk.to(dev), y.to(dev), Rt.to(dev)
            g, R = model(x, ph, mk); loss = 0.0
            if with_geo:
                mu, lv = g[:, :D], g[:, D:].clamp(-6, 4); loss = loss + (0.5 * (lv + (y - mu).square() * torch.exp(-lv))).mean()
            if with_rot:
                loss = loss + (a.w_rot if a.mode == "uni" else 1.0) * geodesic(R, Rt).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss) * len(x); n += len(x)
        ev = evaluate(); hist.append({"epoch": epoch + 1, "train_loss": tot / n, "val_rot_deg": ev.get("rot_deg"), "val_geo_cm": ev["geo"]["vec_err_cm_mean"] if "geo" in ev else None, "score": ev["score"]})
        if ev["score"] < best:
            best, best_ev, best_ep = ev["score"], ev, epoch + 1
            torch.save({"model": model.state_dict(), "fm": fm, "fs": fs, "ym": ym, "ys": ys, "mode": a.mode, "cas": a.mode == "cas"}, ck_dir / f"{name}.pt")
        if (epoch + 1) % 5 == 0 or a.smoke: print(json.dumps(hist[-1]), flush=True)
    res = {"status": "smoke_only" if a.smoke else "formal_validation_only", "mode": a.mode, "seed": a.seed, "rot_deg_prior": a.rot_deg, "scale_range": [lo, hi],
           "n_train": len(ytr), "n_val": len(yva), "no_information_rotation_deg": rot_base, "mean_geometry_baseline_cm": geo_base, "best_epoch": best_ep,
           "val": best_ev, "history": hist, "elapsed_s": time.time() - t0, "test_accessed": False}
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"[{a.mode}] val rotation error {best_ev.get('rot_deg', float('nan')):.2f} deg (no-information {rot_base:.2f}) | geometry error "
          f"{best_ev['geo']['vec_err_cm_mean'] if 'geo' in best_ev else float('nan'):.2f} cm (mean baseline {geo_base:.2f})", flush=True)
    if a.predict:
        stt = torch.load(ck_dir / f"{name}.pt", map_location=dev, weights_only=False); model.load_state_dict(stt["model"])
        for path in a.predict.split(","):
            ex = torch.load(path, weights_only=False); imu_e = ex["imu"].float(); me = E1.valid_mask(ex)
            tgt = E1.targets(ex, prov, "r2"); Rtrue = rot_targets(ex, a.rot_deg, a.lever_cm, a.pred_offset_key)
            if a.ids_scale:
                sl, sh = [float(x) for x in a.ids_scale.split(",")]
                se = torch.tensor([scale_ident(i, sl, sh) for i in ex["identity"]], dtype=torch.float32)
                imu_e, tgt = E1.apply_scale(imu_e, tgt, se)
            if a.mode == "cas":
                imu_e = correct_imu(imu_e, stage1_R(imu_e, ex["phase_id"].long(), me))
            xe = (E1.make_inputs({"imu": imu_e}, "phys", a.stride) - fm) / fs * me[..., None]
            mu_e, lv_e, R_e = predict(xe, ex["phase_id"].long(), me)
            if R_e is not None and "identity" in ex:      # identity-level orientation error (mean of the identity's session estimates, projected to SO(3))
                by = {}
                for k_, ident in enumerate(ex["identity"]):
                    by.setdefault(ident, []).append(k_)
                errs = []
                for ident, idx in by.items():
                    Mm = R_e[idx].mean(0); U_, S_, Vh_ = torch.linalg.svd(Mm); dt_ = torch.det(U_ @ Vh_)
                    Rm = U_ @ torch.diag_embed(torch.stack([torch.ones_like(dt_), torch.ones_like(dt_), dt_], -1)) @ Vh_
                    errs.append(torch.rad2deg(geodesic(Rm, Rtrue[idx[0]])).mean())
                print(f"[{a.mode}] {Path(path).stem}: identity-level orientation error {float(torch.stack(errs).mean()):.3f} deg over {len(errs)} identities "
                      f"(session-level {float(torch.rad2deg(geodesic(R_e, Rtrue)).mean()):.3f})", flush=True)
            torch.save({"identity": ex["identity"], "key": ex["key"], "mu_cm": mu_e, "logvar_cm": lv_e, "target_cm": tgt, "R_est": R_e, "R_true": Rtrue, "mode": a.mode},
                       Path(path).parent / f"pred_{name}_{Path(path).stem}.pt")


if __name__ == "__main__":
    main()
