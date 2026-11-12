"""Calibration sessions for the downstream identities (E8): each identity's own beta and its own installation lever
(offset_key = identity, so the lever equals the one baked into data_local/line14_store_mount3), full nuisance.
usage: python scripts_run/line14_build_cal_identities.py --out data_local/line14_cal_ids_dev10 [--train-sessions 8 --val-sessions 4]"""
import argparse, os, time
from pathlib import Path
os.environ["MOBILEPOSER_FPS"] = "60"
import numpy as np, torch

_M = _C = None
PROTO = os.environ.get("LINE14_PROTO", "slow"); STRIDE = int(os.environ.get("LINE14_STRIDE", "4"))


def _init():
    global _M, _C
    torch.set_num_threads(2)
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    from mobileposer.line9.calib import CONFIG
    from mobileposer.surface_imu import load_attachments
    _M = ParametricModel(paths.smpl_file); _C = load_attachments(CONFIG, _M)


def _job(j):
    from mobileposer.line9.calib import identity_offsets
    from mobileposer.line9.robust_calib import synth_random_calibration
    fold, identity, beta, s, rot, lever_cm, dev, amp, keep = j
    key = f"cal14::id::{identity}::{s}"
    if PROTO == "dyn":
        from mobileposer.line14.protocol import make_dynamic_protocol, synth_random_calibration_p
        out = synth_random_calibration_p(_M, _C, make_dynamic_protocol(), key, beta, identity, rot, lever_cm, dev, stride=STRIDE, amp_scale=amp)
    else:
        out = synth_random_calibration(_M, _C, key, beta, identity, rot, lever_cm, dev)
    _, lever, _ = identity_offsets(identity, rot, lever_cm, gauge=True)
    return {"key": key, "fold": fold, "identity": identity, "session": s, "beta": np.asarray(beta, dtype=np.float32), "lever": lever.numpy(),
            "imu": out["imu"].numpy(), "phase_id": out["phase_id"].numpy(), "amp": float(amp), "keep": np.asarray(keep, dtype=bool)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--train-sessions", type=int, default=8); ap.add_argument("--val-sessions", type=int, default=4)
    ap.add_argument("--rot-deg", type=float, default=3.0); ap.add_argument("--lever-cm", type=float, default=3.0); ap.add_argument("--dev-deg", type=float, default=10.0)
    ap.add_argument("--max-identities", type=int)
    ap.add_argument("--degr", choices=["none", "mix"], default="none"); ap.add_argument("--seed", type=int, default=20261016)
    a = ap.parse_args()
    drng = np.random.default_rng(a.seed)

    def degr(kind):
        amp = float(drng.uniform(0.3, 0.5)) if kind in ("amp", "both") else 1.0
        keep = np.ones(7, dtype=bool)
        if kind in ("skip", "both"):
            keep[:] = False; keep[drng.choice(7, 4, replace=False)] = True
        return amp, keep.tolist()
    from mobileposer.line9.data import load_identity_records
    rec = load_identity_records(folds=("train", "validation"), max_identities_per_fold=a.max_identities)
    ident = {f: {} for f in ("train", "validation")}
    for fold, rows in rec.items():
        for r in rows: ident[fold].setdefault(r.identity, np.asarray(r.beta, dtype=np.float64)[:10])
    jobs = []
    for fold, ns in (("train", a.train_sessions), ("validation", a.val_sessions)):
        for i in sorted(ident[fold]):
            for s in range(ns):
                am, kp = degr(["skip", "amp", "both"][s % 3]) if a.degr == "mix" else (1.0, [True] * 7)
                jobs.append((fold, i, ident[fold][i].tolist(), s, a.rot_deg, a.lever_cm, a.dev_deg, am, kp))
    import multiprocessing as mp
    res = []; t = time.time()
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs)), initializer=_init) as pool:
        for n, o in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            res.append(o)
            if n % 200 == 0: print(n, len(jobs), round(time.time() - t), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)

    def pack(sel):
        sel = sorted(sel, key=lambda o: (o["identity"], o["session"]))
        return {"imu": torch.from_numpy(np.stack([o["imu"] for o in sel])), "phase_id": torch.from_numpy(np.stack([o["phase_id"] for o in sel])),
                "beta": torch.from_numpy(np.stack([o["beta"] for o in sel])), "lever": torch.from_numpy(np.stack([o["lever"] for o in sel])),
                "psi": torch.zeros(len(sel)), "amp": torch.tensor([o["amp"] for o in sel]), "keep": torch.from_numpy(np.stack([o["keep"] for o in sel])),
                "key": [o["key"] for o in sel], "identity": [o["identity"] for o in sel], "session": [o["session"] for o in sel]}
    torch.save(pack([o for o in res if o["fold"] == "train"]), a.out / "train.pt")
    torch.save(pack([o for o in res if o["fold"] == "validation"]), a.out / "val.pt")
    print("done", len(res), round(time.time() - t))
