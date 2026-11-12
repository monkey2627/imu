"""Precomputed per-sequence store for line 9 (IMU input + mean-shape/no-translation FK joints).

Pure caching: values are produced by exactly the same code path as
``PairedWindowDataset._get_sequence`` and stored as float32 memmaps, so training
no longer re-runs full-sequence FK for every window.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


def _index(records) -> tuple[Dict[str, tuple[int, int]], int]:
    index, offset = {}, 0
    for r in sorted(records, key=lambda r: r.key):
        index[r.key] = (offset, r.frames)
        offset += r.frames
    return index, offset


def _worker(args):
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths

    root, cache_path, items, total, shaped = args
    torch.set_num_threads(2)
    root = Path(root)
    imu_mm = np.load(root / "imu.npy", mmap_mode="r+")
    jnt_mm = np.load(root / "joints.npy", mmap_mode="r+")
    body = ParametricModel(paths.smpl_file, device=torch.device("cpu"))
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for row, off, frames, key, beta in items:
            acc = cache["acc"][row].float()[:, :5] / 30.0
            ori = cache["ori"][row].float()[:, :5]
            pose = cache["pose"][row].float()
            assert len(pose) == len(acc) == len(ori) == frames, key
            imu = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
            _, joints = body.forward_kinematics(
                pose=pose.reshape(-1, 216), shape=torch.tensor(beta, dtype=torch.float32) if shaped else None, tran=None, calc_mesh=False)
            imu_mm[off : off + frames] = imu.numpy()
            jnt_mm[off : off + frames] = joints.reshape(-1, 24, 3).numpy()
    imu_mm.flush(); jnt_mm.flush()
    return cache_path, len(items)


def build_store(records: Sequence, root: Path | str, workers: int = 16, shaped: bool = False) -> None:
    import multiprocessing as mp

    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    index, total = _index(records)
    np.lib.format.open_memmap(root / "imu.npy", mode="w+", dtype=np.float32, shape=(total, 60)).flush()
    np.lib.format.open_memmap(root / "joints.npy", mode="w+", dtype=np.float32, shape=(total, 24, 3)).flush()
    by_file = defaultdict(list)
    for r in records:
        by_file[str(r.cache_path)].append((r.row, index[r.key][0], r.frames, r.key, r.beta))
    jobs = [(str(root), p, items, total, shaped) for p, items in by_file.items()]
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        for p, n in pool.imap_unordered(_worker, jobs):
            print("built", p, n, flush=True)
    tmp = root / "index.tmp"
    tmp.write_text(json.dumps({"total_frames": total, "index": index}))
    os.replace(tmp, root / "index.json")


class SequenceStore:
    def __init__(self, root: Path | str):
        root = Path(root)
        meta = json.loads((root / "index.json").read_text())
        self.index = {k: tuple(v) for k, v in meta["index"].items()}
        self.imu = np.load(root / "imu.npy", mmap_mode="r")
        self.joints = np.load(root / "joints.npy", mmap_mode="r")

    def get(self, key: str):
        off, n = self.index[key]
        return (torch.from_numpy(np.array(self.imu[off : off + n])),
                torch.from_numpy(np.array(self.joints[off : off + n])))

    def slice(self, key: str, start: int, stop: int):
        off, n = self.index[key]
        stop = min(stop, n)
        return (torch.from_numpy(np.array(self.imu[off + start : off + stop])),
                torch.from_numpy(np.array(self.joints[off + start : off + stop])))
