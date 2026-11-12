"""Build randomized calibration sessions for line 14 / E6 (virtual subjects).

Each sample = one randomized P0-P6-like calibration (pose-amplitude scaling, time warp, stance deviations, random heading;
mobileposer.line9.robust_calib.synth_random_calibration) on a body of shape beta with an independent installation (per-sample
lever-arm offset and nuisance I2S rotation).

  train : N virtual subjects; beta = a real AMASS *training*-identity beta (prob. p_real) or a draw from the Gaussian fitted to
          the training betas (adult-like population)
  val   : validation identities' real betas (held-out identities), several sessions each
Targets are NOT stored; R1m is recomputed from (beta, lever) with mobileposer.line14.geom (zero pose, T pose).

usage: python scripts_run/line14_build_cal_set.py --out data_local/line14_cal_dev10 [--n-train 4000] [--val-sessions 12]
"""
import argparse, json, os, time
from pathlib import Path

os.environ["MOBILEPOSER_FPS"] = "60"
import numpy as np
import torch

_M = _C = None
PROTO = os.environ.get("LINE14_PROTO", "slow"); STRIDE = int(os.environ.get("LINE14_STRIDE", "4"))


def _init():
    global _M, _C
    torch.set_num_threads(2)
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    from mobileposer.line9.calib import CONFIG
    from mobileposer.surface_imu import load_attachments
    _M = ParametricModel(paths.smpl_file)
    _C = load_attachments(CONFIG, _M)


def _job(j):
    from mobileposer.line9.calib import identity_offsets
    from mobileposer.line9.robust_calib import synth_random_calibration
    key, fold, identity, beta, rot_deg, lever_cm, dev = j[:7]
    amp = j[7] if len(j) > 7 else 1.0
    keep = list(j[8]) if len(j) > 8 else [True] * 7
    if PROTO == "dyn":
        from mobileposer.line14.protocol import make_dynamic_protocol, synth_random_calibration_p
        out = synth_random_calibration_p(_M, _C, make_dynamic_protocol(), key, beta, key, rot_deg, lever_cm, dev, stride=STRIDE, amp_scale=amp)
    else:
        out = synth_random_calibration(_M, _C, key, beta, key, rot_deg, lever_cm, dev)
    _, lever, _ = identity_offsets(key, rot_deg, lever_cm, gauge=True)
    return {"key": key, "fold": fold, "identity": identity, "dev": float(dev), "beta": np.asarray(beta, dtype=np.float32), "lever": lever.numpy(),
            "imu": out["imu"].numpy(), "phase_id": out["phase_id"].numpy(), "psi": out["psi"], "amp": float(amp), "keep": np.asarray(keep, dtype=bool)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=4000); ap.add_argument("--val-sessions", type=int, default=12)
    ap.add_argument("--p-real", type=float, default=0.5); ap.add_argument("--rot-deg", type=float, default=3.0)
    ap.add_argument("--lever-cm", type=float, default=3.0); ap.add_argument("--dev-deg", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20261015); ap.add_argument("--dev-mix", default=""); ap.add_argument("--degr-mix", action="store_true"); ap.add_argument("--max-identities", type=int)
    a = ap.parse_args()
    from mobileposer.line9.data import load_identity_records
    rec = load_identity_records(folds=("train", "validation"), max_identities_per_fold=a.max_identities)
    ident = {f: {} for f in ("train", "validation")}
    for fold, rows in rec.items():
        for r in rows:
            ident[fold].setdefault(r.identity, np.asarray(r.beta, dtype=np.float64)[:10])
    tr_ids = sorted(ident["train"]); tr_b = np.stack([ident["train"][i] for i in tr_ids])
    mu, cov = tr_b.mean(0), np.cov(tr_b.T)
    rng = np.random.default_rng(a.seed)

    def draw_degr(r, kind=None):
        """(amp_scale, keep[7]): none 40% | skip (keep 4 of 7 phases) 20% | tiny motions (x0.3-0.5) 20% | both 20%."""
        kind = kind or ["none", "skip", "amp", "both"][int(r.choice(4, p=[0.4, 0.2, 0.2, 0.2]))]
        amp = float(r.uniform(0.3, 0.5)) if kind in ("amp", "both") else 1.0
        keep = np.ones(7, dtype=bool)
        if kind in ("skip", "both"):
            keep[:] = False; keep[r.choice(7, 4, replace=False)] = True
        return amp, keep.tolist()
    jobs = []
    for n in range(a.n_train):
        if rng.random() < a.p_real:
            k = int(rng.integers(len(tr_ids))); beta, ident_name = tr_b[k], tr_ids[k]
        else:
            beta, ident_name = rng.multivariate_normal(mu, cov), "gauss"
        dev_n = float(rng.choice([float(x) for x in a.dev_mix.split(",")])) if a.dev_mix else a.dev_deg
        extra = draw_degr(rng) if a.degr_mix else (1.0, [True] * 7)
        jobs.append((f"cal14::train::{n}", "train", ident_name, beta.tolist(), a.rot_deg, a.lever_cm, dev_n, extra[0], extra[1]))
    for i in sorted(ident["validation"]):
        for s in range(a.val_sessions):
            dev_v = [float(x) for x in a.dev_mix.split(",")][s % len(a.dev_mix.split(","))] if a.dev_mix else a.dev_deg
            extra = draw_degr(rng, ["none", "skip", "amp", "both"][s % 4]) if a.degr_mix else (1.0, [True] * 7)
            jobs.append((f"cal14::val::{i}::{s}", "validation", i, ident["validation"][i].tolist(), a.rot_deg, a.lever_cm, dev_v, extra[0], extra[1]))
    import multiprocessing as mp
    res = []; t = time.time()
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs)), initializer=_init) as pool:
        for n, o in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            res.append(o)
            if n % 200 == 0: print(n, len(jobs), round(time.time() - t), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)

    def pack(sel):
        sel = sorted(sel, key=lambda o: o["key"])
        return {"imu": torch.from_numpy(np.stack([o["imu"] for o in sel])), "phase_id": torch.from_numpy(np.stack([o["phase_id"] for o in sel])),
                "beta": torch.from_numpy(np.stack([o["beta"] for o in sel])), "lever": torch.from_numpy(np.stack([o["lever"] for o in sel])),
                "psi": torch.tensor([o["psi"] for o in sel]), "dev": torch.tensor([o["dev"] for o in sel]),
                "amp": torch.tensor([o["amp"] for o in sel]), "keep": torch.from_numpy(np.stack([o["keep"] for o in sel])),
                "key": [o["key"] for o in sel], "identity": [o["identity"] for o in sel]}
    torch.save(pack([o for o in res if o["fold"] == "train"]), a.out / "train.pt")
    torch.save(pack([o for o in res if o["fold"] == "validation"]), a.out / "val.pt")
    (a.out / "meta.json").write_text(json.dumps(vars(a), default=str) + "\n")
    print("done", len(res), round(time.time() - t))
