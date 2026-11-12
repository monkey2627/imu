"""Calibration-Oracle data (line 9): per-identity fixed sensor-to-segment rotation and lever-arm offsets.

IMU synthesis re-uses the production surface synthesis (``mobileposer.no_head_layouts``, 60 fps,
PNP surface attachments).  Per identity and sensor, one fixed rotation ``R_off`` (measured
orientation = R_bone @ R_off) and one fixed lever arm ``l`` in the bone frame (sensor point =
attachment point + R_bone @ l) are shared by every motion of that identity.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MOBILEPOSER_FPS", "60")
LABELS = ["lw", "rw", "ls", "rs", "pelvis"]
CONFIG = "mobileposer/configs/pnp_surface_imu.json"


def identity_offsets(identity: str, rot_deg: float, lever_cm: float, data_seed: int = 20261013, gauge=None):
    """Deterministic per-identity (R_off [5,3,3], lever [5,3] metres); rot_deg is the RMS angle."""
    digest = hashlib.sha256(f"{identity}|{data_seed}".encode()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
    rotvec = torch.randn(5, 3, generator=g, dtype=torch.float64) * (np.deg2rad(rot_deg) / np.sqrt(3.0))
    lever = torch.randn(5, 3, generator=g, dtype=torch.float64)
    lever = lever / lever.norm(dim=-1, keepdim=True).clamp_min(1e-9) * (lever_cm / 100.0)
    from mobileposer.articulate import math as M
    R = M.axis_angle_to_rotation_matrix(rotvec.float()).reshape(5, 3, 3)
    if gauge is None:
        gauge = os.environ.get("LINE9_GAUGE") == "1"
    if gauge:
        # Heading gauge: the pelvis sensor's own yaw offset is indistinguishable from the (unknown) global heading,
        # so it is defined away: world-yaw component of the pelvis offset is removed.
        a = torch.atan2(R[4, 0, 2], R[4, 2, 2])
        c, sn = torch.cos(-a), torch.sin(-a)
        Y = torch.tensor([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]], dtype=R.dtype)
        R = R.clone(); R[4] = Y @ R[4]
    return R, lever.float(), rotvec.float()


def calibration_vector(identity, rot_deg, lever_cm, data_seed=20261013):
    _, lever, rotvec = identity_offsets(identity, rot_deg, lever_cm, data_seed)
    return torch.cat((rotvec.reshape(-1), lever.reshape(-1)))  # 15 + 15


@torch.no_grad()
def synth_sequence(model, config, pose, shape, tran, R_off=None, lever=None, chunk_size=128, drift=None):
    from mobileposer.no_head_layouts import _syn_acc
    from mobileposer.surface_imu import surface_kinematics
    rot, _, pts = surface_kinematics(model, pose, shape, tran, LABELS, config, chunk_size)
    rot, pts = rot[:, :5], pts[:, :5]
    if R_off is not None:
        pts = pts + (rot @ lever[None, :, :, None]).squeeze(-1)
        rot = rot @ R_off[None]
        if drift is not None:
            rot = rot @ drift
    acc = _syn_acc(pts)
    return acc, rot


def make_drift(n: int, drift_deg: float, key: str, fps: int = 60, data_seed: int = 20261013):
    """Slow random-walk I2S drift per sensor: RMS angle = drift_deg after 60 s (grows as sqrt(t))."""
    from mobileposer.articulate import math as M
    digest = hashlib.sha256(f"{key}|drift|{data_seed}".encode()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
    step = np.deg2rad(drift_deg) / np.sqrt(3.0) / np.sqrt(60.0 * fps)
    w = torch.cumsum(torch.randn(n, 5, 3, generator=g, dtype=torch.float64) * step, dim=0).float()
    return M.axis_angle_to_rotation_matrix(w.reshape(-1, 3)).reshape(n, 5, 3, 3)


def _worker(args):
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    from mobileposer.surface_imu import load_attachments
    root, cache_path, items, rot_deg, lever_cm = args[:5]
    drift_deg = args[5] if len(args) > 5 else 0.0
    torch.set_num_threads(2)
    model = ParametricModel(paths.smpl_file)
    config = load_attachments(CONFIG, model)
    imu_mm = np.load(Path(root) / "imu.npy", mmap_mode="r+")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_offsets = {}
    for row, off, frames, key, identity in items:
        if identity not in cache_offsets:
            cache_offsets[identity] = identity_offsets(identity, rot_deg, lever_cm)[:2]
        R, l = cache_offsets[identity]
        pose = cache["pose"][row].float(); shape = cache["shape"][row].float(); tran = cache["tran"][row].float()
        drift = make_drift(frames, drift_deg, key) if drift_deg > 0 else None
        acc, ori = synth_sequence(model, config, pose, shape, tran, R, l, drift=drift)
        assert len(acc) == frames, key
        imu_mm[off:off + frames] = torch.cat((acc.flatten(1) / 30.0, ori.flatten(1)), dim=1).numpy()
    imu_mm.flush()
    return cache_path, len(items)


def build_calib_store(records, shaped_store: Path, root: Path, rot_deg: float, lever_cm: float, workers=24, piece_frames=120_000, drift_deg: float = 0.0):
    import multiprocessing as mp
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(shaped_store) / "index.json").read_text())
    total = meta["total_frames"]
    np.lib.format.open_memmap(root / "imu.npy", mode="w+", dtype=np.float32, shape=(total, 60)).flush()
    for name in ("joints.npy",):
        link = root / name
        if not link.exists():
            os.symlink((Path(shaped_store) / name).resolve(), link)
    by_file = defaultdict(list)
    for r in records:
        by_file[str(r.cache_path)].append((r.row, meta["index"][r.key][0], r.frames, r.key, r.identity))
    jobs = []
    for p, items in by_file.items():
        piece, n = [], 0
        for it in items:
            piece.append(it); n += it[2]
            if n >= piece_frames:
                jobs.append((str(root), p, piece, rot_deg, lever_cm, drift_deg)); piece, n = [], 0
        if piece:
            jobs.append((str(root), p, piece, rot_deg, lever_cm, drift_deg))
    jobs.sort(key=lambda j: -sum(i[2] for i in j[2]))
    print("jobs", len(jobs), flush=True)
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        for p, n in pool.imap_unordered(_worker, jobs):
            print("built", Path(p).name, n, flush=True)
    (root / "index.json").write_text(json.dumps(meta))
    (root / "calib.json").write_text(json.dumps({"rot_deg_rms": rot_deg, "lever_cm": lever_cm, "drift_deg_rms_per_min": drift_deg, "data_seed": 20261013}))
