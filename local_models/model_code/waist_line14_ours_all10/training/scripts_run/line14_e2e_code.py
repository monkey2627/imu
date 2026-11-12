"""Line 14: task-supervised PERSONAL CODE from the calibration session, trained end-to-end against the pose (angle) loss.

  calibration session --(phys features)--> CodeEncoder --> z (zdim, zero-initialised)
  geometry g = PivotPrint estp6 (as before) -----------------------------+
  FiLM Joints(imu, g, z)  ->  joints  ->  Poser(cat(joints, imu), z)  ->  pose
  loss = pose_mse(+temporal) + joint_w * joint_mse        (same as line14_e2e.py)

Everything starts from the end-to-end film model of C5 (e2e_film, same seed), so z=0 reproduces it exactly.  Arms:
  --arm code   encoder + FiLM branches for z are trained together with Joints and Poser
  --arm ctrl   identical schedule, no code (continued training of the same model)  -- budget control
Evaluation (validation identities only): SIP / Ang / Pos / Mesh with the real code, with SHUFFLED codes (identity i gets the code of another identity),
and with z = 0.  A genuine personal effect needs real < ctrl and real < shuffled.
usage: python scripts_run/line14_e2e_code.py --arm code --seed 42 --gpu 0 --tag C15
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import mobileposer.articulate as art
from mobileposer.config import paths
from mobileposer.line9.data import PairedWindowDataset, load_identity_records
from mobileposer.line9.models import GeometryStats
from mobileposer.line9.enc2 import phys_features
from mobileposer.line9.robust_calib import heading_normalize
from mobileposer.line14.geom import GeomProvider
from mobileposer.line14.models import FilmJoints
from types import SimpleNamespace
import line14_poser_chain as C
from line14_estimate_r1 import R1Encoder


class FilmJointsZ(FilmJoints):
    """FilmJoints plus a zero-initialised FiLM branch driven by the personal code z (input layer and output layer)."""
    def __init__(self, geom_dim, stats, zdim):
        super().__init__(geom_dim, stats)
        self.geom_dim_g, self.zdim = geom_dim, zdim
        self.film_in_z = nn.Linear(zdim, 2 * self.hidden_dim); self.film_out_z = nn.Linear(zdim, 2 * self.output_dim)
        for m in (self.film_in_z, self.film_out_z):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, imu, input_lengths=None, geometry=None):      # geometry = [g (6) | z (zdim)]
        g_part, z = geometry[:, :self.geom_dim_g], geometry[:, self.geom_dim_g:]
        c = self._conditioning(imu, g_part)
        data = self.linear1(imu)
        g, b = self.film_in(c).chunk(2, dim=-1); gz, bz = self.film_in_z(z.to(c.dtype)).chunk(2, dim=-1)
        data = data * (1.0 + g[:, None, :] + gz[:, None, :]) + (b + bz)[:, None, :]
        data = self.dropout(F.relu(data))
        if input_lengths is not None:
            data = pack_padded_sequence(data, torch.as_tensor(input_lengths, dtype=torch.long, device="cpu"), batch_first=True, enforce_sorted=False)
        data, _ = self.rnn(data)
        if input_lengths is not None:
            data, _ = pad_packed_sequence(data, batch_first=True)
        out = self.linear2(data)
        go, bo = self.film_out(c).chunk(2, dim=-1); goz, boz = self.film_out_z(z.to(c.dtype)).chunk(2, dim=-1)
        return out * (1.0 + go[:, None, :] + goz[:, None, :]) + (bo + boz)[:, None, :]


class PoserZ(C.ChainPoser):
    """ChainPoser whose hidden FiLM is driven by the last zdim entries of the [g | z] vector."""
    def __init__(self, zdim, gdim):
        super().__init__(132, zdim, GeometryStats(torch.zeros(zdim), torch.ones(zdim)))
        self.gdim = gdim

    def forward(self, x, lengths=None, geom=None):
        return super().forward(x, lengths, geom[:, self.gdim:])


class CodeProvider:
    """Stands in for GeomProvider: returns [estp6 (6) | z (zdim)] for a batch.  Used by line14_poser_chain.evaluate for both Joints and Poser."""
    def __init__(self, base, sessions, encoder, device, zdim, mode="real", partner=None):
        self.base, self.S, self.enc, self.dev, self.zdim, self.mode, self.partner = base, sessions, encoder, device, zdim, mode, partner
        self.dim = base.dim + zdim

    def _pick(self, ident, fixed_seed):
        idx = self.S["by_id"][ident]
        if fixed_seed is None:
            return idx[int(torch.randint(len(idx), (1,)))]
        return idx[int.from_bytes(hashlib.sha256(f"{ident}|{fixed_seed}|code".encode()).digest()[:4], "little") % len(idx)]

    def code(self, identities, fixed_seed):
        rows = [self._pick(self.partner.get(i, i) if self.mode == "shuffle" else i, fixed_seed) for i in identities]
        x = self.S["x"][rows].to(self.dev); ph = self.S["ph"][rows].to(self.dev); mk = self.S["mk"][rows].to(self.dev)
        z = self.enc(x, ph, mk)[:, :self.zdim].float()
        return torch.zeros_like(z) if self.mode == "zero" else z

    def batch(self, betas, identities=None, mode="none", sigma=0.0, fixed_seed=None):
        g = self.base.batch(betas, identities, mode, sigma, fixed_seed).to(self.dev)
        return torch.cat((g, self.code(identities, fixed_seed)), dim=-1)


def load_sessions(split, fm=None, fs=None, stride=2):
    d = torch.load(f"data_local/line14_cal_ids_dyn13/{split}.pt", weights_only=False)
    x = phys_features(heading_normalize(d["imu"].float()), dt=stride / 60.0)
    if fm is None:
        fm = x.mean((0, 1)); fs = x.std((0, 1)).clamp_min(1e-4)
    x = (x - fm) / fs
    by_id = {}
    for i, ident in enumerate(d["identity"]):
        by_id.setdefault(ident, []).append(i)
    return {"x": x, "ph": d["phase_id"].long(), "mk": torch.ones_like(d["phase_id"], dtype=torch.bool), "by_id": by_id}, fm, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["code", "ctrl"], required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="C15"); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr-code", type=float, default=5e-4); ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--zdim", type=int, default=8); ap.add_argument("--z-noise", type=float, default=0.0, help="std of Gaussian noise added to z in training")
    ap.add_argument("--joint-w", type=float, default=1.0); ap.add_argument("--init", default="C5/e2e_film"); ap.add_argument("--window", type=int, default=125)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 1
    C.seed_all(a.seed); device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    kw = dict(max_identities_per_fold=3, max_sequences_per_identity=3) if a.smoke else {}
    rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("train", "validation"), **kw)
    store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
    mk = lambda r: PairedWindowDataset(r, a.window, store=store, file_cache_size=2, sequence_cache_size=64, max_windows=16 if a.smoke else None)
    train, val = mk(rec["train"]), mk(rec["validation"])
    body = art.model.ParametricModel(paths.smpl_file, device=device); parent = body.parent
    from mobileposer.evaluate_standard import StdEvaluator
    ev = StdEvaluator(device)
    base = GeomProvider("estp6", 3.0, "data_local/line14_est_r2_b90.pt")
    S_tr, fm, fs = load_sessions("train"); S_va, _, _ = load_sessions("val", fm, fs)
    ck = torch.load(f"checkpoints/line14/{a.init}_s{a.seed}.pt", map_location="cpu", weights_only=False)
    ckj = torch.load(f"checkpoints/line14/E18/estp6_oracle_film_s{a.seed}.pt", map_location="cpu", weights_only=False)
    jnet = FilmJointsZ(6, GeometryStats(ckj["geometry_mean"], ckj["geometry_std"]), a.zdim)
    print("load jnet:", jnet.load_state_dict(ck["jnet"], strict=False).missing_keys)
    poser = PoserZ(a.zdim, 6); print("load poser:", poser.load_state_dict(ck["poser"], strict=False).missing_keys)
    jnet.to(device); poser.to(device)
    enc = R1Encoder(S_tr["x"].shape[-1], dim=a.zdim).to(device)
    nn.init.normal_(enc.out.weight, std=0.1)          # NOT zero: z-FiLM is zero-initialised, two zero layers in a row would never train
    prov_tr = CodeProvider(base, S_tr, enc, device, a.zdim, "real"); prov_va = CodeProvider(base, S_va, enc, device, a.zdim, "real")
    ids_va = sorted(S_va["by_id"]); prov_va.partner = {ids_va[i]: ids_va[(i + len(ids_va) // 2) % len(ids_va)] for i in range(len(ids_va))}
    td, ts = C.loader(train, a.batch, a.seed, True); vd, _ = C.loader(val, a.batch, a.seed, False)
    groups = [{"params": list(jnet.parameters()) + list(poser.parameters()), "lr": a.lr}]
    if a.arm == "code":
        groups.append({"params": list(enc.parameters()), "lr": a.lr_code})
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    name = f"{a.arm}_z{a.zdim}" + (f"_n{a.z_noise:g}" if a.z_noise else "") + f"_s{a.seed}"
    out = Path("results/line14") / a.tag / f"{name}.json"; ckp = Path("checkpoints/line14") / a.tag / f"{name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True); ckp.parent.mkdir(parents=True, exist_ok=True)

    def step_inputs(b, train_mode, prov):
        imu, jt = C.get_inputs(b, device); L = b["target_lengths"].tolist()
        g = base.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=None if train_mode else a.seed).to(device)
        if a.arm == "code":
            z = prov.code(b["identity"], None if train_mode else a.seed)
            if train_mode and a.z_noise > 0:
                z = z + a.z_noise * torch.randn_like(z)
        else:
            z = torch.zeros(len(g), a.zdim, device=device)
        return imu, jt, L, torch.cat((g, z), dim=-1)

    def val_score():
        poser.eval(); jnet.eval(); enc.eval(); ve = vn = 0; je = jn = 0
        with torch.no_grad():
            for b in vd:
                mask = b["target_mask"].to(device)
                imu, jt, L, gz = step_inputs(b, False, prov_va)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    j = jnet(imu, L, gz); p = poser(torch.cat((j.to(imu.dtype), imu), -1), L, gz)
                y = C.global_targets(b["target_joints"][..., 3:].to(device), parent)
                ve += float(((p.float() - y).square().mean(-1) * mask).sum()); vn += int(mask.sum())
                je += float(((j.float() - jt).reshape(*mask.shape, 24, 3).norm(dim=-1).mean(-1) * mask).sum()) * 100; jn += int(mask.sum())
        return ve / vn, je / jn

    best, hist, t0 = float("inf"), [], time.time()
    s0, j0 = val_score(); print(json.dumps({"epoch": 0, "val_mse_6d": s0, "joint_err_cm": j0}), flush=True)
    for epoch in range(a.epochs):
        ts.set_epoch(epoch); poser.train(); jnet.train(); enc.train(); tot = n = 0
        for b in td:
            mask = b["target_mask"].to(device)
            with torch.no_grad():
                y = C.global_targets(b["target_joints"][..., 3:].to(device), parent)
            opt.zero_grad(set_to_none=True)
            imu, jt, L, gz = step_inputs(b, True, prov_tr)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                j = jnet(imu, L, gz); p = poser(torch.cat((j.to(imu.dtype), imu), -1), L, gz)
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
            best = s; torch.save({"poser": poser.state_dict(), "jnet": jnet.state_dict(), "enc": enc.state_dict(), "epoch": epoch + 1, "score": s}, ckp, pickle_protocol=4)
    st = torch.load(ckp, map_location=device, weights_only=False); poser.load_state_dict(st["poser"]); jnet.load_state_dict(st["jnet"]); enc.load_state_dict(st["enc"])
    args = SimpleNamespace(variant="joints", seed=a.seed)
    metrics = {}
    for mode in (["real", "shuffle", "zero"] if a.arm == "code" else ["zero"]):
        prov_va.mode = mode
        metrics[mode] = C.evaluate(poser, vd, args, device, body, ev, prov_va, prov_va, "film", None, jnet, a.seed)
        print(json.dumps({"run": name, "mode": mode, **{m: round(metrics[mode][m], 3) for m in ("sip", "ang", "pos", "mesh")}}), flush=True)
    prov_va.mode = "real"; _, je = val_score()
    res = {"status": "smoke_only" if a.smoke else "formal_validation_only", "arm": a.arm, "seed": a.seed, "zdim": a.zdim, "z_noise": a.z_noise,
           "protocol": {"epochs": a.epochs, "batch": a.batch, "lr": a.lr, "lr_code": a.lr_code, "init": a.init, "window": a.window,
                        "target_window_digest_validation": val.target_window_digest(), "test_accessed": False},
           "best_epoch": st["epoch"], "joint_err_cm_final": je, "val_mse_before": s0, "metrics": metrics, "history": hist, "elapsed_seconds": time.time() - t0}
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)


if __name__ == "__main__":
    main()
