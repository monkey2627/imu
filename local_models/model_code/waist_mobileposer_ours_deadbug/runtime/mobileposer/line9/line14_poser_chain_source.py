"""Line 14 chain experiment (plan C): geometry -> Joints(FiLM) -> Poser -> SIP / Ang (validation-only, synthetic).

A Poser-lite (same 2-layer BiLSTM as the Joints replica; output = 16 reduced joints x 6D GLOBAL rotation, as the public Poser) is trained
on the AMASS train identities with the public Poser's input convention: IMU (60) + joint positions (72).  As in the public Poser
(models/poser.py::_joint_noise) the joints fed in TRAINING are the ground-truth individual-shape joints plus i.i.d. noise (--jnoise m);
at test time they are supplied by a frozen Joints module.  One Poser is trained per (variant, seed) and evaluated with several joint sources:

  --variant imu         input IMU only (no Joints input)
  --variant joints      input IMU + joints, no geometry
  --variant joints_geo  input IMU + joints + FiLM(geometry) on the Poser itself (--geom estp6 | height | pca6 ...)

joint sources at evaluation (frozen Joints, same seed):  gt (oracle GT joints, noise-free) | b0 (Joints without geometry, E5 r1_b0) |
film (FilmJoints with the estimated PivotPrint estp6, E18).  Metrics: SIP / Ang (global orientation) and Pos / Mesh of the standard
evaluator (StdEvaluator: pose-only FK with the MEAN body), identity-averaged; validation identities only, AMASS test / TotalCapture untouched.
Run from the line-14 workspace root with PYTHONPATH=$PWD and LINE9_GAUGE unset.
"""
from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

import mobileposer.articulate as art
from mobileposer.articulate import math as AM
from mobileposer.config import joint_set, paths
from mobileposer.line9.data import PairedWindowDataset, SubsetBatchSampler, collate_paired_windows, load_identity_records
from mobileposer.line9.models import GeometryStats, Line9Joints, fit_geometry_stats
from mobileposer.line9.store import SequenceStore
from mobileposer.line14.geom import GeomProvider
from mobileposer.line14.models import FilmJoints
from mobileposer.utils.model_utils import reduced_pose_to_full

REDUCED = list(joint_set.reduced)


class ChainStore:
    """IMU + shape-aware joints from a SequenceStore, local 6D pose from pose6d.npy; slice() returns joints(3) ++ pose6d(6) per joint."""

    def __init__(self, imu_root, pose_root):
        self.base = SequenceStore(imu_root)
        meta = json.loads((Path(pose_root) / "index.json").read_text())
        assert {k: tuple(v) for k, v in meta["index"].items()} == self.base.index, "pose store index differs from IMU store index"
        self.index = self.base.index
        self.pose = np.load(Path(pose_root) / "pose6d.npy", mmap_mode="r")

    def slice(self, key, start, stop):
        off, n = self.index[key]
        stop = min(stop, n)
        imu = torch.from_numpy(np.array(self.base.imu[off + start:off + stop]))
        j = torch.from_numpy(np.array(self.base.joints[off + start:off + stop]))
        p = torch.from_numpy(np.array(self.pose[off + start:off + stop]))
        return imu, torch.cat((j, p), dim=-1)

    def get(self, key):
        return self.slice(key, 0, self.index[key][1])


class ChainPoser(nn.Module):
    """Linear(in,256) -> [FiLM] -> ReLU/Dropout -> 2-layer BiLSTM(256) -> Linear(512,96); FiLM (zero-init) on the 256 hidden channels."""

    def __init__(self, in_dim, geom_dim=0, stats: GeometryStats | None = None):
        super().__init__()
        self.geom_dim = geom_dim
        self.linear1 = nn.Linear(in_dim, 256)
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True)
        self.linear2 = nn.Linear(512, 96)
        self.dropout = nn.Dropout(0.4)
        if geom_dim:
            self.film = nn.Linear(geom_dim, 512)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
            self.register_buffer("gmean", stats.mean.clone()); self.register_buffer("gstd", stats.std.clone())

    def forward(self, x, lengths=None, geom=None):
        h = self.linear1(x)
        if self.geom_dim:
            c = (geom.to(h.dtype) - self.gmean.to(h.dtype)) / self.gstd.to(h.dtype)
            g, b = self.film(c).chunk(2, dim=-1)
            h = h * (1 + g[:, None]) + b[:, None]
        h = self.dropout(F.relu(h))
        if lengths is not None:
            h = pack_padded_sequence(h, torch.as_tensor(lengths, dtype=torch.long, device="cpu"), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(h)
        if lengths is not None:
            h, _ = pad_packed_sequence(h, batch_first=True)
        return self.linear2(h)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def loader(ds, batch, seed, shuffle):
    sampler = SubsetBatchSampler(ds, batch, shuffle=shuffle, drop_last=False, seed=seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_paired_windows, num_workers=0, pin_memory=True), sampler


def global_targets(pose6d, parent):
    B, T = pose6d.shape[:2]
    Rl = AM.r6d_to_rotation_matrix(pose6d.reshape(-1, 6)).reshape(B * T, 24, 3, 3)
    Rg = AM.forward_kinematics_R(Rl, parent).reshape(B, T, 24, 3, 3)
    return AM.rotation_matrix_to_r6d(Rg[:, :, REDUCED].reshape(-1, 3, 3)).reshape(B, T, 16 * 6)


CFG = {"hook": None, "sr": None, "b0": "checkpoints/line14/E5/r1_b0_s{seed}.pt", "film": "checkpoints/line14/E18/estp6_oracle_film_s{seed}.pt"}


def get_inputs(b, device):
    """IMU [B,T,60] and GT joints [B,T,72] (individual-shape FK); with CFG["sr"]=(lo,hi) the wide-population scaling of line 14 E19 is applied
    (accelerations and joint positions x s(identity), s in [lo,hi]; orientations / pose targets unchanged)."""
    imu = b["target_imu"].to(device); jt = b["target_joints"][..., :3].reshape(*imu.shape[:2], 72).to(device)
    if CFG["sr"]:
        from mobileposer.line14.geom import scale_ident
        s = torch.tensor([scale_ident(i, *CFG["sr"]) for i in b["identity"]], dtype=torch.float32, device=device)
        imu = imu.clone(); imu[..., :15] = imu[..., :15] * s[:, None, None]; jt = jt * s[:, None, None]
    if CFG.get("hook") is not None:
        imu = CFG["hook"](imu, b)
    return imu, jt


def load_frozen(seed, device):
    """Frozen B0 Joints (E5 r1_b0) and FiLM-estp6 Joints (E18) of the same seed (paths from CFG)."""
    ck = torch.load(CFG["b0"].format(seed=seed), map_location="cpu", weights_only=False)
    b0 = Line9Joints("b0", geom_dim=len(ck["geometry_mean"]), geometry_stats=GeometryStats(ck["geometry_mean"], ck["geometry_std"]))
    b0.load_state_dict(ck["model"]); b0.to(device).eval()
    ck = torch.load(CFG["film"].format(seed=seed), map_location="cpu", weights_only=False)
    film = FilmJoints(len(ck["geometry_mean"]), GeometryStats(ck["geometry_mean"], ck["geometry_std"]))
    film.load_state_dict(ck["model"]); film.to(device).eval()
    return b0, film


@torch.no_grad()
def joint_source(src, b, imu, joints_gt, geom_est, b0, film, device):
    if src == "gt":
        return joints_gt
    if src.startswith("gtn"):       # diagnostic: GT joints + i.i.d. noise of std <src[3:]> cm (fixed generator)
        g = torch.Generator().manual_seed(123)
        return joints_gt + float(src[3:]) / 100.0 * torch.randn(joints_gt.shape, generator=g).to(joints_gt.device)
    L = b["target_lengths"].tolist()
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        p = b0(imu, L) if src == "b0" else film(imu, L, geom_est)
    return p.float()


@torch.no_grad()
def evaluate(model, dl, args, device, body, ev, prov_est, prov_in, src, b0, film, noise_seed):
    model.eval(); per = {}; sc = sn = 0
    for b in dl:
        imu, jt = get_inputs(b, device); mask = b["target_mask"].to(device)
        pose6 = b["target_joints"][..., 3:].to(device)
        geom_est = prov_est.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=args.seed).to(device)
        geom_in = prov_in.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=args.seed).to(device) if prov_in else None
        jin = joint_source(src, b, imu, jt, geom_est, b0, film, device) if args.variant != "imu" else None
        x = imu if jin is None else torch.cat((jin, imu), dim=-1)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = model(x, b["target_lengths"].tolist(), geom_in).float()
        p = p.reshape(*p.shape[:2], 16, 6)
        Rr = AM.r6d_to_rotation_matrix(p.reshape(-1, 6)).reshape(*p.shape[:2], 16, 3, 3)
        for k in range(len(p)):
            n = int(b["target_mask"][k].sum())
            full = torch.eye(3, device=device).repeat(n, 24, 1, 1); full[:, REDUCED] = Rr[k, :n]
            pl = body.inverse_kinematics_R(full)
            for ig in joint_set.ignored:
                pl[:, ig] = torch.eye(3, device=device)
            gl = AM.r6d_to_rotation_matrix(pose6[k, :n].reshape(-1, 6)).reshape(n, 24, 3, 3)
            e = ev(pl, gl)
            d = per.setdefault(b["identity"][k], [0.0, 0.0, 0.0, 0.0, 0])
            for q in range(4):
                d[q] += float(e[q]) * n
            d[4] += n
    ids = sorted(per)
    res = {i: dict(zip(("sip", "ang", "pos", "mesh"), [per[i][q] / per[i][4] for q in range(4)])) for i in ids}
    return {"per_identity": res, **{m: float(np.mean([res[i][m] for i in ids])) for m in ("sip", "ang", "pos", "mesh")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["imu", "joints", "joints_geo"], required=True)
    ap.add_argument("--geom", default="estp6", help="geometry fed to the Poser FiLM (joints_geo): estp6 | height | pca6 | ...")
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="C1"); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--weight-decay", type=float, default=0.01); ap.add_argument("--window", type=int, default=125)
    ap.add_argument("--jnoise", type=float, default=0.04, help="std (m) of the training noise added to the GT joints (public Poser default 0.04)")
    ap.add_argument("--imu-store", type=Path, default=Path("data_local/line14_store_mount3")); ap.add_argument("--pose-store", type=Path, default=Path("data/line9_store_pose6d"))
    ap.add_argument("--est-file", default="data_local/line14_est_r2_b90.pt")
    ap.add_argument("--fk-loss", choices=["none", "mean"], default="none", help="mean = public-Poser FK joint-position loss (mean body)")
    ap.add_argument("--train-joints", choices=["gt_noise", "pred_b0", "pred_film"], default="gt_noise", help="joint input used to TRAIN the Poser")
    ap.add_argument("--pred-noise", type=float, default=0.0, help="extra i.i.d. noise (m) on predicted joints during training")
    ap.add_argument("--scale-range", default="", help="wide-population scaling lo,hi (line 14 E19); selects the E19 frozen Joints unless overridden")
    ap.add_argument("--b0-ckpt", default=None); ap.add_argument("--film-ckpt", default=None); ap.add_argument("--film-geom", default="estp6")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 1
    if a.scale_range:
        CFG["sr"] = tuple(float(x) for x in a.scale_range.split(","))
        CFG["b0"] = a.b0_ckpt or "checkpoints/line14/E19/r1_b0_s{seed}.pt"; CFG["film"] = a.film_ckpt or f"checkpoints/line14/E19/{a.film_geom}_oracle_film_s{{seed}}.pt"
    seed_all(a.seed); device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    kw = dict(max_identities_per_fold=3, max_sequences_per_identity=3) if a.smoke else {}
    rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("train", "validation"), **kw)
    store = ChainStore(a.imu_store, a.pose_store)
    mk = lambda r: PairedWindowDataset(r, a.window, store=store, file_cache_size=2, sequence_cache_size=64, max_windows=16 if a.smoke else None)
    train, val = mk(rec["train"]), mk(rec["validation"])
    body = art.model.ParametricModel(paths.smpl_file, device=device); parent = body.parent
    from mobileposer.evaluate_standard import StdEvaluator
    ev = StdEvaluator(device)
    kind = a.film_geom; prov_est = GeomProvider(kind, 3.0, a.est_file if "estp" in kind else None, None if "estp" in kind else CFG["sr"])   # geometry for the frozen FiLM Joints
    prov_in, stats = None, None
    if a.variant == "joints_geo":
        prov_in = prov_est if a.geom == "estp6" else GeomProvider(a.geom, 3.0, a.est_file)
        rows, seen = [], set()
        for r in rec["train"]:
            if r.identity not in seen:
                beta, _ = train._get_geometry(r); rows.append(prov_in.base(beta, r.identity)); seen.add(r.identity)
        stats = fit_geometry_stats(torch.stack(rows))
    in_dim = 60 if a.variant == "imu" else 132
    model = ChainPoser(in_dim, prov_in.dim if prov_in else 0, stats).to(device)
    b0, film = load_frozen(a.seed, device)
    td, ts = loader(train, a.batch, a.seed, True); vd, _ = loader(val, a.batch, a.seed, False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    name = f"{a.variant}" + (f"_{a.geom}" if a.variant == "joints_geo" else "") + (f"_{a.train_joints}" if a.train_joints != "gt_noise" else "") + f"_s{a.seed}"
    out = Path("results/line14") / a.tag / f"{name}.json"; ck = Path("checkpoints/line14") / a.tag / f"{name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True); ck.parent.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(20261021 + a.seed)

    def make_x(b, train_mode):
        imu, j = get_inputs(b, device)
        if a.variant == "imu":
            return imu, None
        if a.train_joints == "gt_noise":
            if train_mode:
                j = j + a.jnoise * torch.randn(j.shape, device=device)
            else:
                j = j + a.jnoise * torch.randn(j.shape, generator=gen).to(device)
        else:           # train on the frozen Joints module's own predictions (+ optional extra noise): the error distribution seen at test time
            ge = prov_est.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=None if train_mode else a.seed).to(device)
            j = joint_source(a.train_joints[5:], b, imu, j, ge, b0, film, device)
            if train_mode and a.pred_noise > 0:
                j = j + a.pred_noise * torch.randn(j.shape, device=device)
        g = prov_in.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=None if train_mode else a.seed).to(device) if prov_in else None
        return torch.cat((j, imu), -1), g

    best, hist, t0 = float("inf"), [], time.time()
    for epoch in range(a.epochs):
        ts.set_epoch(epoch); model.train(); tot = n = 0
        for b in td:
            x, g = make_x(b, True); mask = b["target_mask"].to(device)
            with torch.no_grad():
                y = global_targets(b["target_joints"][..., 3:].to(device), parent)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(x, b["target_lengths"].tolist(), g)
                mse = ((pred.float() - y).square() * mask[:, :, None]).sum() / (mask.sum() * 96)
                acc = pred[:, 2:] + pred[:, :-2] - 2 * pred[:, 1:-1]; tm = mask[:, 2:] & mask[:, :-2] & mask[:, 1:-1]
                loss = mse + 1e-5 * (acc.abs() * tm[:, :, None]).sum() / tm.sum().clamp_min(1)
            if a.fk_loss == "mean":     # public Poser: FK of the predicted pose with the MEAN body vs the (individual-shape) target joints
                Bq, Sq = pred.shape[:2]
                rp = AM.r6d_to_rotation_matrix(pred.float().reshape(-1, 6)).reshape(-1, 16, 3, 3)
                full = reduced_pose_to_full(rp.unsqueeze(0)).squeeze(0).reshape(-1, 24, 3, 3)
                local = body.inverse_kinematics_R(full)
                for ig in joint_set.ignored:
                    local[:, ig] = torch.eye(3, device=device)
                local[:, 0] = full[:, 0]
                jp = body.forward_kinematics(pose=local.reshape(-1, 216))[1].reshape(Bq, Sq, 72)
                jt = b["target_joints"][..., :3].reshape(Bq, Sq, 72).to(device)
                loss = loss + (((jp - jt).square().mean(-1)) * mask).sum() / mask.sum()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss) * int(mask.sum()); n += int(mask.sum())
        model.eval(); ve = vn = 0; gen.manual_seed(20261021 + a.seed)
        with torch.no_grad():
            for b in vd:
                x, g = make_x(b, False); mask = b["target_mask"].to(device)
                y = global_targets(b["target_joints"][..., 3:].to(device), parent)
                pred = model(x, b["target_lengths"].tolist(), g).float()
                ve += float(((pred - y).square().mean(-1) * mask).sum()); vn += int(mask.sum())
        score = ve / vn
        hist.append({"epoch": epoch + 1, "train_loss": tot / max(n, 1), "val_mse_6d": score}); print(json.dumps({"variant": name, **hist[-1]}), flush=True)
        if score < best:
            best = score; torch.save({"model": model.state_dict(), "epoch": epoch + 1, "score": score}, ck, pickle_protocol=4)
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
    srcs = ["none"] if a.variant == "imu" else ["gt", "b0", "film"]
    metrics = {s: evaluate(model, vd, a, device, body, ev, prov_est, prov_in, s, b0, film, a.seed) for s in srcs}
    res = {"status": "smoke_only" if a.smoke else "formal_validation_only", "variant": a.variant, "geom": a.geom if a.variant == "joints_geo" else None, "seed": a.seed,
           "protocol": {"epochs": a.epochs, "batch": a.batch, "window": a.window, "jnoise": a.jnoise, "imu_store": str(a.imu_store), "est_file": a.est_file,
                        "target_window_digest_validation": val.target_window_digest(), "test_accessed": False},
           "best_epoch": int(np.argmin([h["val_mse_6d"] for h in hist])) + 1, "metrics": metrics, "history": hist, "elapsed_seconds": time.time() - t0}
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)
    print(json.dumps({"variant": name, **{s: {m: round(metrics[s][m], 3) for m in ("sip", "ang", "pos", "mesh")} for s in srcs}}))


if __name__ == "__main__":
    main()
