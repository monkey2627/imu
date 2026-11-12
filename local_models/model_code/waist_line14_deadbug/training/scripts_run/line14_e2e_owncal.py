"""Line 14 (C18): can a short sample of the SAME PERSON'S OWN motion (a different sequence of the identity, as in the line-9 loader) yield a personal code
that improves the rotation estimate?  Redesign of C15, which was inconclusive because (a) the synthetic calibration session contains no personal motion
style by construction and (b) the whole network was fine-tuned, over-fitted by epoch 3 and the code stayed constant.

  calibration sample (IMU only, another sequence of the identity, Tc frames) -> subsample -> heading-normalise -> physical features -> CodeEncoder -> z
  Joints(imu, estp6, z) -> joints -> Poser(cat(joints, imu), z) -> pose            [Joints and Poser are the C5 e2e_film model, FROZEN]
  only the encoder and the zero/small-initialised z-FiLM layers are trained, loss = pose_mse(+temporal) + joint_mse.
Control = the unchanged C5 e2e_film (z not used) and the SAME trained model evaluated with z = 0 and with codes swapped between identities.
usage: python scripts_run/line14_e2e_owncal.py --seed 42 --gpu 0 --tag C18 --cal-frames 600
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import mobileposer.articulate as art
from mobileposer.config import paths
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line9.models import GeometryStats
from mobileposer.line9.enc2 import phys_features
from mobileposer.line9.robust_calib import heading_normalize
from mobileposer.line14.geom import GeomProvider
from types import SimpleNamespace
import line14_poser_chain as C
import line14_e2e_code as E
from line14_estimate_r1 import R1Encoder

STRIDE = 4


def cal_features(imu, lengths):
    """imu [B,T,60] (60 fps, acc/30 | ori) -> features [B,T',135] (sub-sampled by STRIDE), mask [B,T']"""
    x = heading_normalize(imu[:, ::STRIDE].contiguous().float())
    f = phys_features(x, dt=STRIDE / 60.0)
    f = F.layer_norm(f, f.shape[-1:])
    Tn = f.shape[1]
    mk = torch.arange(Tn, device=f.device)[None] < ((torch.as_tensor(lengths, device=f.device) + STRIDE - 1) // STRIDE)[:, None]
    return f * mk[..., None], mk


class CalBank:
    """identity -> calibration IMU tensor, taken from the dataset's own (cross-sequence) calibration pairs."""
    def __init__(self, ds):
        self.imu = {}
        seen = set()
        for i, w in enumerate(ds.windows):
            ident = w.target.identity
            if ident in seen:
                continue
            seen.add(ident); self.imu[ident] = ds[i]["calibration_imu"]
        self.ids = sorted(self.imu)

    def batch(self, identities, device):
        xs = [self.imu[i] for i in identities]; L = [len(x) for x in xs]; T = max(L)
        out = torch.zeros(len(xs), T, 60)
        for k, x in enumerate(xs):
            out[k, :len(x)] = x
        return out.to(device), L


class OwnCalProvider:
    """Stand-in for GeomProvider in line14_poser_chain.evaluate: returns [estp6 (6) | z (zdim)]."""
    def __init__(self, base, bank, enc, device, zdim, mode="real", partner=None):
        self.base, self.bank, self.enc, self.dev, self.zdim, self.mode, self.partner = base, bank, enc, device, zdim, mode, partner or {}

    def code(self, identities):
        ids = [self.partner.get(i, i) if self.mode == "shuffle" else i for i in identities]
        imu, L = self.bank.batch(ids, self.dev)
        f, mk = cal_features(imu, L)
        z = self.enc(f, torch.zeros(mk.shape, dtype=torch.long, device=self.dev), mk)[:, :self.zdim].float()
        return torch.zeros_like(z) if self.mode == "zero" else z

    def batch(self, betas, identities=None, mode="none", sigma=0.0, fixed_seed=None):
        g = self.base.batch(betas, identities, mode, sigma, fixed_seed).to(self.dev)
        return torch.cat((g, self.code(identities)), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--tag", default="C18")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01); ap.add_argument("--zdim", type=int, default=8); ap.add_argument("--cal-frames", type=int, default=600)
    ap.add_argument("--joint-w", type=float, default=1.0); ap.add_argument("--init", default="C5/e2e_film"); ap.add_argument("--window", type=int, default=125)
    ap.add_argument("--unfreeze", action="store_true", help="also fine-tune Joints and Poser (lr 1e-4)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 1
    C.seed_all(a.seed); device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    kw = dict(max_identities_per_fold=3, max_sequences_per_identity=3) if a.smoke else {}
    rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("train", "validation"), **kw)
    store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
    mk = lambda r: PairedWindowDataset(r, a.window, calibration_window_size=a.cal_frames, store=store, file_cache_size=2, sequence_cache_size=64, max_windows=16 if a.smoke else None)
    train, val = mk(rec["train"]), mk(rec["validation"])
    body = art.model.ParametricModel(paths.smpl_file, device=device); parent = body.parent
    from mobileposer.evaluate_standard import StdEvaluator
    ev = StdEvaluator(device)
    base = GeomProvider("estp6", 3.0, "data_local/line14_est_r2_b90.pt")
    ck = torch.load(f"checkpoints/line14/{a.init}_s{a.seed}.pt", map_location="cpu", weights_only=False)
    ckj = torch.load(f"checkpoints/line14/E18/estp6_oracle_film_s{a.seed}.pt", map_location="cpu", weights_only=False)
    jnet = E.FilmJointsZ(6, GeometryStats(ckj["geometry_mean"], ckj["geometry_std"]), a.zdim); jnet.load_state_dict(ck["jnet"], strict=False)
    poser = E.PoserZ(a.zdim, 6); poser.load_state_dict(ck["poser"], strict=False)
    jnet.to(device); poser.to(device)
    for m in (jnet.film_in_z, jnet.film_out_z, poser.film):          # small random (not zero) so that the encoder receives gradient from step 1
        nn.init.normal_(m.weight, std=0.002); nn.init.zeros_(m.bias)
    enc = R1Encoder(135, dim=a.zdim).to(device); nn.init.normal_(enc.out.weight, std=0.1)
    train_params = list(enc.parameters()) + list(jnet.film_in_z.parameters()) + list(jnet.film_out_z.parameters()) + list(poser.film.parameters())
    ids_new = {id(p) for p in train_params}
    if not a.unfreeze:
        for p in list(jnet.parameters()) + list(poser.parameters()):
            p.requires_grad_(id(p) in ids_new)
        groups = [{"params": train_params, "lr": a.lr}]
    else:
        rest = [p for p in list(jnet.parameters()) + list(poser.parameters()) if id(p) not in ids_new]
        groups = [{"params": train_params, "lr": a.lr}, {"params": rest, "lr": 1e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    bank_va = CalBank(val); ids_va = bank_va.ids
    prov_va = OwnCalProvider(base, bank_va, enc, device, a.zdim, "real", {ids_va[i]: ids_va[(i + len(ids_va) // 2) % len(ids_va)] for i in range(len(ids_va))})
    td, ts = C.loader(train, a.batch, a.seed, True); vd, _ = C.loader(val, a.batch, a.seed, False)
    name = f"owncal_z{a.zdim}_c{a.cal_frames}" + ("_unfrz" if a.unfreeze else "") + f"_s{a.seed}"
    out = Path("results/line14") / a.tag / f"{name}.json"; ckp = Path("checkpoints/line14") / a.tag / f"{name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True); ckp.parent.mkdir(parents=True, exist_ok=True)

    def z_train(b):
        imu = b["calibration_imu"].to(device); f, m = cal_features(imu, b["calibration_lengths"].tolist())
        return enc(f, torch.zeros(m.shape, dtype=torch.long, device=device), m)[:, :a.zdim].float()

    def val_score(mode="real"):
        poser.eval(); jnet.eval(); enc.eval(); prov_va.mode = mode; ve = vn = 0; je = jn = 0
        with torch.no_grad():
            for b in vd:
                mask = b["target_mask"].to(device); imu, jt = C.get_inputs(b, device); L = b["target_lengths"].tolist()
                gz = prov_va.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=a.seed)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    j = jnet(imu, L, gz); p = poser(torch.cat((j.to(imu.dtype), imu), -1), L, gz)
                y = C.global_targets(b["target_joints"][..., 3:].to(device), parent)
                ve += float(((p.float() - y).square().mean(-1) * mask).sum()); vn += int(mask.sum())
                je += float(((j.float() - jt).reshape(*mask.shape, 24, 3).norm(dim=-1).mean(-1) * mask).sum()) * 100; jn += int(mask.sum())
        prov_va.mode = "real"
        return ve / vn, je / jn

    best, hist, t0 = float("inf"), [], time.time()
    s0, j0 = val_score("zero"); sr0, jr0 = val_score("real"); print(json.dumps({"epoch": 0, "val_mse_6d(z=0)": s0, "joint_err_cm(z=0)": j0, "val_mse_6d(real z at init)": sr0, "joint_err_cm(real z at init)": jr0}), flush=True)
    for epoch in range(a.epochs):
        ts.set_epoch(epoch); poser.train(); jnet.train(); enc.train(); tot = n = 0
        for b in td:
            mask = b["target_mask"].to(device); imu, jt = C.get_inputs(b, device); L = b["target_lengths"].tolist()
            with torch.no_grad():
                y = C.global_targets(b["target_joints"][..., 3:].to(device), parent)
                g = base.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=None).to(device)
            opt.zero_grad(set_to_none=True)
            gz = torch.cat((g, z_train(b)), dim=-1)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                j = jnet(imu, L, gz); p = poser(torch.cat((j.to(imu.dtype), imu), -1), L, gz)
                mse = ((p.float() - y).square() * mask[:, :, None]).sum() / (mask.sum() * 96)
                acc = p[:, 2:] + p[:, :-2] - 2 * p[:, 1:-1]; tm = mask[:, 2:] & mask[:, :-2] & mask[:, 1:-1]
                loss = mse + 1e-5 * (acc.abs() * tm[:, :, None]).sum() / tm.sum().clamp_min(1)
                if a.joint_w > 0:
                    loss = loss + a.joint_w * ((j.float() - jt).square() * mask[:, :, None]).sum() / (mask.sum() * 72)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(train_params, 1.0); scaler.step(opt); scaler.update()
            tot += float(loss) * int(mask.sum()); n += int(mask.sum())
        s, je = val_score(); sz, _ = val_score("zero"); sh, _ = val_score("shuffle")
        hist.append({"epoch": epoch + 1, "train_loss": tot / max(n, 1), "val_mse_6d": s, "val_mse_6d_zero": sz, "val_mse_6d_shuffle": sh, "joint_err_cm": je}); print(json.dumps({"run": name, **hist[-1]}), flush=True)
        if s < best:
            best = s; torch.save({"poser": poser.state_dict(), "jnet": jnet.state_dict(), "enc": enc.state_dict(), "epoch": epoch + 1, "score": s}, ckp, pickle_protocol=4)
    st = torch.load(ckp, map_location=device, weights_only=False); poser.load_state_dict(st["poser"]); jnet.load_state_dict(st["jnet"]); enc.load_state_dict(st["enc"])
    args = SimpleNamespace(variant="joints", seed=a.seed); metrics = {}
    enc.eval()
    for mode in ("real", "shuffle", "zero"):
        prov_va.mode = mode
        metrics[mode] = C.evaluate(poser, vd, args, device, body, ev, prov_va, prov_va, "film", None, jnet, a.seed)
        print(json.dumps({"run": name, "mode": mode, **{m: round(metrics[mode][m], 3) for m in ("sip", "ang", "pos", "mesh")}}), flush=True)
    prov_va.mode = "real"; _, je = val_score()
    res = {"status": "smoke_only" if a.smoke else "formal_validation_only", "seed": a.seed, "zdim": a.zdim, "cal_frames": a.cal_frames, "unfreeze": a.unfreeze,
           "protocol": {"epochs": a.epochs, "batch": a.batch, "lr": a.lr, "init": a.init, "window": a.window, "target_window_digest_validation": val.target_window_digest(), "test_accessed": False},
           "best_epoch": st["epoch"], "joint_err_cm_final": je, "val_mse_zero_start": s0, "metrics": metrics, "history": hist, "elapsed_seconds": time.time() - t0}
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)


if __name__ == "__main__":
    main()
