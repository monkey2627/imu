"""Local-pose (6D) targets for the Poser-lite angle experiments of line 9.

``build_pose_store`` writes ``pose6d.npy`` [frames, 24, 6] (local rotations of the cached AMASS pose, same frame order as the
line-9 sequence stores).  ``PoseStore`` pairs it with any IMU store (``imu.npy`` + ``index.json``) so that
``PairedWindowDataset`` returns the local 6D pose as ``target_joints``.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def _worker(args):
    from mobileposer.articulate import math as AM
    root, cache_path, items = args
    torch.set_num_threads(2)
    mm = np.load(Path(root) / "pose6d.npy", mmap_mode="r+")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    for row, off, frames in items:
        pose = cache["pose"][row].float().reshape(-1, 3, 3)
        assert len(pose) == frames * 24
        mm[off:off + frames] = AM.rotation_matrix_to_r6d(pose).reshape(frames, 24, 6).numpy()
    mm.flush()
    return cache_path, len(items)


def build_pose_store(records: Sequence, shaped_store: Path | str, root: Path | str, workers: int = 16) -> None:
    import multiprocessing as mp
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(shaped_store) / "index.json").read_text())
    np.lib.format.open_memmap(root / "pose6d.npy", mode="w+", dtype=np.float32, shape=(meta["total_frames"], 24, 6)).flush()
    by_file = defaultdict(list)
    for r in records:
        by_file[str(r.cache_path)].append((r.row, meta["index"][r.key][0], r.frames))
    jobs = [(str(root), p, items) for p, items in by_file.items()]
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        for p, n in pool.imap_unordered(_worker, jobs):
            print("built", Path(p).name, n, flush=True)
    (root / "index.json").write_text(json.dumps(meta))


class PoseStore:
    """IMU from ``imu_root`` (imu.npy + index.json), local 6D pose from ``pose_root`` (pose6d.npy)."""

    def __init__(self, imu_root: Path | str, pose_root: Path | str):
        imu_root, pose_root = Path(imu_root), Path(pose_root)
        meta = json.loads((imu_root / "index.json").read_text())
        self.index = {k: tuple(v) for k, v in meta["index"].items()}
        self.imu = np.load(imu_root / "imu.npy", mmap_mode="r")
        self.pose = np.load(pose_root / "pose6d.npy", mmap_mode="r")

    def get(self, key: str):
        off, n = self.index[key]
        return torch.from_numpy(np.array(self.imu[off:off + n])), torch.from_numpy(np.array(self.pose[off:off + n]))

    def slice(self, key: str, start: int, stop: int):
        off, n = self.index[key]
        stop = min(stop, n)
        return (torch.from_numpy(np.array(self.imu[off + start:off + stop])),
                torch.from_numpy(np.array(self.pose[off + start:off + stop])))
