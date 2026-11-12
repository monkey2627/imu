"""E6: estimate the R1m geometric representation (12-d) and its uncertainty from a randomized calibration session.

Input : calibration IMU (heading-normalised), 15 Hz, 555 frames, phase ids P0..P6  (data_local/line14_cal_*/{train,val}.pt)
Target: R1m = R1(beta) + (lever_k - lever_pelvis)  [cm], 4 vectors (lw, rw, ls, rs)  -- zero pose / SMPL axes
Model : physics-feature BiGRU + per-phase attention encoder (mobileposer.line9.enc2.FeatEncoder structure) with a 24-d head:
        12 normalised means + 12 log-variances, trained with a Gaussian NLL (diagonal Sigma-hat).
Reports on held-out validation identities: error per vector (cm, % of vector norm, in units of the between-person SD),
        the train-mean baseline, and calibration of Sigma-hat (|err| vs sigma-hat correlation, 1-sigma / 2-sigma coverage).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mobileposer.line9.enc2 import FeatEncoder, phys_features
from mobileposer.line9.robust_calib import heading_normalize
from mobileposer.line14.geom import GeomProvider, scale_ident


class TFWrap(nn.Module):
    """Drop-in for the BiGRU of FeatEncoder: learned positions + TransformerEncoder, output width 2*hidden."""
    def __init__(self, hidden, layers=4, heads=4, ff=256, dropout=0.1, max_len=1024):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_len, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(hidden, 2 * hidden)

    def forward(self, f):
        return self.out(self.enc(f + self.pos[:, : f.shape[1]])), None


class R1Encoder(FeatEncoder):
    def __init__(self, in_dim, dim=12, enc="gru", **kw):
        super().__init__(in_dim, **kw)
        if enc == "tf":
            self.gru = TFWrap(self.proj[0].out_features)
        latent = self.norm.normalized_shape[0]
        self.dim = dim
        self.out = nn.Linear(latent, 2 * dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, phase_ids, mask=None):
        if mask is None:
            mask = torch.ones_like(phase_ids, dtype=torch.bool)
        f = self.proj(x) + self.phase_emb(phase_ids.masked_fill(~mask, 0))
        enc, _ = self.gru(f)
        safe = phase_ids.masked_fill(~mask, 0)
        member = (safe[:, None, :] == torch.arange(self.max_phases, device=x.device)[None, :, None]) & mask[:, None, :]
        pm = member.any(-1)
        score = self.frame_att(enc).squeeze(-1)[:, None, :].masked_fill(~member, -torch.inf).masked_fill(~pm[:, :, None], 0.0)
        w = torch.softmax(score, -1) * member.to(score.dtype)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        tokens = self.phase_proj(torch.einsum("bpt,btd->bpd", w, enc))
        pooled, _ = self.att(self.query.expand(x.shape[0], -1, -1), tokens, tokens, key_padding_mask=~pm, need_weights=False)
        return self.out(self.norm(pooled[:, 0]))


def targets(blob, prov, target="r1m"):
    rows = []
    for beta, lever in zip(blob["beta"], blob["lever"]):
        if target == "r1m":
            r1 = prov.table.get(beta).reshape(4, 3)
            rows.append((r1 + (lever[:4] - lever[4:5])).reshape(12))
        else:   # r2m: sensor -> joint vectors including the installation offset;  r2: without
            from mobileposer.line14.geom import R2_PAIRS
            pts, joints = prov.table.points_joints(beta)
            if target == "r2m": pts = pts + lever
            rows.append(torch.stack([joints[j] - pts[k] for k, j in R2_PAIRS]).reshape(24))
    return torch.stack(rows) * 100.0          # cm


def make_inputs(blob, feat, stride=4):
    imu = heading_normalize(blob["imu"].float())
    return phys_features(imu, dt=stride / 60.0) if feat == "phys" else imu


def valid_mask(blob):
    """[N,T] bool: frames that belong to a phase the (possibly degraded) session actually performed."""
    ph = blob["phase_id"].long()
    if "keep" in blob:
        return blob["keep"].bool().gather(1, ph)
    return torch.ones_like(ph, dtype=torch.bool)


def scale_for(keys, lo, hi, seed):
    import hashlib
    out = []
    for k in keys:
        h = int.from_bytes(hashlib.sha256(f"{k}|{seed}|scale".encode()).digest()[:8], "little") / 2**64
        out.append(lo + (hi - lo) * h)
    return torch.tensor(out, dtype=torch.float32)


def apply_scale(imu_raw, y, s):
    """Global body scaling s: accelerations scale with s (acc channels = first 15 dims), orientations unchanged; R1m target scales with s."""
    imu_raw = imu_raw.clone(); imu_raw[..., :15] *= s[:, None, None]
    return imu_raw, y * s[:, None]


def metrics(mu, y, logvar, ystd):
    nv = y.shape[1] // 3
    err = (mu - y).reshape(-1, nv, 3)
    e = err.norm(dim=-1)                                          # [N,nvec] cm
    ynorm = y.reshape(-1, nv, 3).norm(dim=-1)
    sig = torch.exp(0.5 * logvar).reshape(-1, nv, 3)
    sig_vec = sig.norm(dim=-1)
    zabs = (err.abs() / sig)
    def rank(a): return a.argsort(-1).argsort(-1).float()
    corr = float(np.corrcoef(rank(sig_vec.flatten()).numpy(), rank(e.flatten()).numpy())[0, 1])
    return {"vec_err_cm": e.mean(0).tolist(), "vec_err_cm_mean": float(e.mean()), "vec_err_pct_of_norm": float((e / ynorm).mean() * 100),
            "err_over_population_sd": float((err.abs() / ystd.reshape(1, nv, 3)).mean()),
            "sigma_vec_cm_mean": float(sig_vec.mean()), "spearman_sigma_vs_err": corr,
            "coverage_1sigma": float((zabs < 1).float().mean()), "coverage_2sigma": float((zabs < 2).float().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data_local/line14_cal_dev10")); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--feat", choices=["phys", "raw"], default="phys")
    ap.add_argument("--target", choices=["r1m", "r2m", "r2"], default="r1m"); ap.add_argument("--exclude-ids", default=""); ap.add_argument("--enc", choices=["gru", "tf"], default="gru"); ap.add_argument("--stride", type=int, default=4); ap.add_argument("--predict", default=""); ap.add_argument("--pred-name", default=""); ap.add_argument("--scale-range", default="1,1"); ap.add_argument("--ids-scale", default=""); ap.add_argument("--tag", default="E6a"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    lo, hi = [float(x) for x in a.scale_range.split(",")]
    prov = GeomProvider("r1")
    tr = torch.load(a.data / "train.pt", weights_only=False); va = torch.load(a.data / "val.pt", weights_only=False)
    if a.smoke:
        for d in (tr, va):
            for k in ("imu", "phase_id", "beta", "lever", "psi", "keep", "amp"):
                if k in d: d[k] = d[k][:16]
            for k in ("key", "identity"): d[k] = d[k][:16]
        a.epochs = 1
    if a.exclude_ids:
        excl = set(json.load(open(a.exclude_ids)))
        keep = torch.tensor([i not in excl for i in tr["identity"]])
        for k in ("imu", "phase_id", "beta", "lever", "psi", "keep", "amp"):
            if k in tr: tr[k] = tr[k][keep]
        for k in ("key", "identity"): tr[k] = [x for x, kp in zip(tr[k], keep.tolist()) if kp]
        print("excluded identities:", len(excl), "-> training samples", int(keep.sum()), flush=True)
    ytr, yva = targets(tr, prov, a.target), targets(va, prov, a.target)
    s_tr = scale_for(tr["key"], lo, hi, a.seed) if hi > lo else torch.ones(len(ytr)); s_va = scale_for(va["key"], lo, hi, 7) if hi > lo else torch.ones(len(yva))
    imu_tr, ytr = apply_scale(tr["imu"].float(), ytr, s_tr); imu_va, yva = apply_scale(va["imu"].float(), yva, s_va)
    mtr, mva = valid_mask(tr), valid_mask(va)
    xtr = make_inputs({"imu": imu_tr}, a.feat, a.stride); xva = make_inputs({"imu": imu_va}, a.feat, a.stride)
    fm = (xtr * mtr[..., None]).sum((0, 1)) / mtr.sum().clamp_min(1)
    fs = (((xtr - fm) ** 2 * mtr[..., None]).sum((0, 1)) / mtr.sum().clamp_min(1)).sqrt().clamp_min(1e-4)
    xtr = (xtr - fm) / fs * mtr[..., None]; xva = (xva - fm) / fs * mva[..., None]
    ym, ys = ytr.mean(0), ytr.std(0).clamp_min(1e-3)
    ytr_n = (ytr - ym) / ys
    ph_tr, ph_va = tr["phase_id"].long(), va["phase_id"].long()
    model = R1Encoder(xtr.shape[-1], dim=ytr.shape[1], enc=a.enc).to(device)
    D = ytr.shape[1]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(xtr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1))
    dl = DataLoader(TensorDataset(xtr, ph_tr, mtr, ytr_n), batch_size=a.batch, shuffle=True, drop_last=False)
    out_dir = Path("results/line14") / a.tag; out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir = Path("checkpoints/line14") / a.tag; ck_dir.mkdir(parents=True, exist_ok=True)
    name = f"{a.feat}_{a.target}{'_tf' if a.enc == 'tf' else ''}_s{a.seed}" + ("" if hi == lo else f"_sc{lo}-{hi}") + (f"_{a.pred_name}" if a.pred_name else "")
    base = {"vec_err_cm_mean": None}
    # mean baseline (train mean prediction)
    mu0 = ym[None].expand_as(yva); base = metrics(mu0, yva, torch.zeros_like(yva) + 3.0, ys)
    best, hist, t0 = float("inf"), [], time.time()

    def predict(x, ph, mk):
        model.eval(); mus, lvs = [], []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                o = model(x[i:i + 64].to(device), ph[i:i + 64].to(device), mk[i:i + 64].to(device)).float().cpu()
                mus.append(o[:, :D]); lvs.append(o[:, D:].clamp(-6, 4))
        mu = torch.cat(mus); lv = torch.cat(lvs)
        return mu * ys + ym, lv + 2 * torch.log(ys)      # de-normalise: mean, log-variance in cm^2
    for epoch in range(a.epochs):
        model.train(); tot = n = 0
        for x, ph, mk, y in dl:
            x, ph, mk, y = x.to(device), ph.to(device), mk.to(device), y.to(device)
            o = model(x, ph, mk); mu, lv = o[:, :D], o[:, D:].clamp(-6, 4)
            loss = (0.5 * (lv + (y - mu).square() * torch.exp(-lv))).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss) * len(x); n += len(x)
        mu_v, lv_v = predict(xva, ph_va, mva); m = metrics(mu_v, yva, lv_v, ys)
        hist.append({"epoch": epoch + 1, "train_nll": tot / n, "val_vec_err_cm": m["vec_err_cm_mean"]})
        if m["vec_err_cm_mean"] < best:
            best = m["vec_err_cm_mean"]; best_m = m; best_ep = epoch + 1
            torch.save({"model": model.state_dict(), "fm": fm, "fs": fs, "ym": ym, "ys": ys, "feat": a.feat}, ck_dir / f"{name}.pt")
        if (epoch + 1) % 5 == 0 or a.smoke: print(json.dumps(hist[-1]), flush=True)
    res = {"status": "smoke_only" if a.smoke else "formal_validation_only", "tag": a.tag, "target": a.target, "feat": a.feat, "seed": a.seed, "scale_range": [lo, hi],
           "n_train": len(ytr), "n_val": len(yva), "val_identities": len(set(va["identity"])), "best_epoch": best_ep,
           "target_population_sd_cm": ys.reshape(-1, 3).tolist(), "mean_baseline": base, "model": best_m, "history": hist, "elapsed_s": time.time() - t0, "test_accessed": False}
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    if a.predict:   # predict (cm) on extra sets with the *best* checkpoint
        st = torch.load(ck_dir / f"{name}.pt", map_location=device, weights_only=False); model.load_state_dict(st["model"])
        for path in a.predict.split(","):
            ex = torch.load(path, weights_only=False); tgt = targets(ex, prov, a.target); imu_e = ex["imu"].float()
            if a.ids_scale:
                sl, sh = [float(x) for x in a.ids_scale.split(",")]
                se = torch.tensor([scale_ident(i, sl, sh) for i in ex["identity"]], dtype=torch.float32)
                imu_e, tgt = apply_scale(imu_e, tgt, se)
            me = valid_mask(ex)
            xe = (make_inputs({"imu": imu_e}, a.feat, a.stride) - fm) / fs * me[..., None]
            mu_e, lv_e = predict(xe, ex["phase_id"].long(), me)
            torch.save({"identity": ex["identity"], "key": ex["key"], "mu_cm": mu_e, "logvar_cm": lv_e, "target_cm": tgt, "target": a.target},
                       Path(path).parent / f"pred_{a.pred_name or name}_{Path(path).stem}.pt")
    print("mean baseline: err %.2f cm (%.1f%% of norm)" % (base["vec_err_cm_mean"], base["vec_err_pct_of_norm"]))
    print("model        : err %.2f cm (%.1f%% of norm)  err/SD %.2f  sigma %.2f cm  spearman %.2f  cov1s %.2f cov2s %.2f" % (
        best_m["vec_err_cm_mean"], best_m["vec_err_pct_of_norm"], best_m["err_over_population_sd"], best_m["sigma_vec_cm_mean"],
        best_m["spearman_sigma_vs_err"], best_m["coverage_1sigma"], best_m["coverage_2sigma"]))


if __name__ == "__main__":
    main()
