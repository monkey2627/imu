"""Extract streaming backbone tokens (one per 16-frame grid) for every processed motion.

For each sequence the cache stores, unwindowed: imu6 [T,5,6], calibrated ori [T,5,3,3],
local 6D targets [T,138] and tokens [K,3840] for grid frames g = 15, 31, ... (see
stream.common). The frozen backbone runs in float32, exactly as the online adapter does.

    python -m stream.prepare_stream --source-root <processed>/wrists_shanks_waist \
        --datasets ACCAD BirdDog --out data/features/stream_wristankle_v1 --device cuda:0
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from downstream.data import group_fold, rotation_targets, to_imu6
from downstream.model import encode_imu
from downstream.prepare import BIRDDOG_RAW, bird_dog_groups
from model_factory import imuformer_huge_patch16
from .common import TOKEN_DIM, WINDOW, grid_frames, window_bounds

LABELS = ["lw", "rw", "ls", "rs", "waist"]


@torch.inference_mode()
def encode_windows(backbone, imu, grids, device, batch):
    """imu [T,5,6] -> tokens [len(grids), 3840] fp16 (last patch of each window)."""
    out = []
    for start in range(0, len(grids), batch):
        chunk = grids[start:start + batch].tolist()
        windows = imu.new_zeros((len(chunk), WINDOW, 5, 6))
        masks = torch.zeros((len(chunk), WINDOW), dtype=torch.bool)
        for row, g in enumerate(chunk):
            lo, hi = window_bounds(g)
            src = imu[max(lo, 0):hi]
            windows[row, WINDOW - len(src):] = src
            masks[row, WINDOW - len(src):] = True
        tokens = encode_imu(backbone, windows.to(device), masks.to(device), fps=60)  # fp32, as online
        out.append(tokens[:, -1].float().cpu())
    return torch.cat(out) if out else torch.zeros((0, TOKEN_DIM))


def prepare_one(path, out_path, backbone, device, attachment, batch):
    source = torch.load(path, map_location="cpu", weights_only=False)
    syn = source["synthesis"]
    if (source["layout_labels"] != LABELS or syn["mode"] != "surface" or syn["fps"] != 60
            or syn["attachment_sha256"] != attachment):
        raise ValueError(f"Incompatible synthesis: {path}")
    n = len(source["pose"])
    if "groups" in source:            # e.g. BirdDogMix: explicit groups, train-only
        groups = source["groups"]
    elif path.stem == "BirdDog":
        groups = bird_dog_groups(path, BIRDDOG_RAW)
    else:
        groups = [f"{path.stem}:{i}" for i in range(n)]
    seqs, frames = [], 0
    for i, (acc, ori, pose, group) in enumerate(zip(source["acc"], source["ori"], source["pose"], groups)):
        if len(pose) < 64:
            continue
        imu = to_imu6(acc, ori, fps=60)
        target = rotation_targets(pose)
        if not (torch.isfinite(imu).all() and torch.isfinite(target).all()):
            raise ValueError(f"non-finite {path.stem}:{i}")
        tokens = encode_windows(backbone, imu, grid_frames(len(pose)), device, batch)
        seqs.append({"imu": imu.float(), "ori": ori[:, :5].float(), "target": target.to(torch.float16),
                     "tokens": tokens, "group": group,
                     "val": False if source.get("train_only") else group_fold(group) == "val"})
        frames += len(pose)
        if (i + 1) % 250 == 0:
            print(f"[stream-prepare] {path.stem} {i + 1}/{n}", flush=True)
    report = {"dataset": path.stem, "sequences": len(seqs), "frames": frames,
              "val_sequences": sum(s["val"] for s in seqs)}
    payload = {"sequences": seqs, "metadata": {"source": str(path), "attachment_sha256": attachment,
                                               "sensor_order": LABELS, "fps": 60, "window": WINDOW,
                                               "schedule": "stream.common v1", "precision": "float32",
                                               "report": report}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    print(f"[stream-prepared] {json.dumps(report)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--attachment-sha256", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    device = torch.device(args.device)
    backbone = imuformer_huge_patch16()
    backbone.load_state_dict(torch.load(Path(__file__).resolve().parents[1] / "imudino.pt",
                                        map_location="cpu", weights_only=True), strict=True)
    backbone.eval().requires_grad_(False).to(device)
    for name in args.datasets:
        out = args.out / f"{name}.stream.pt"
        if out.exists():
            print(f"[skip] {out}", flush=True)
            continue
        t0 = time.time()
        prepare_one(args.source_root / f"{name}.pt", out, backbone, device, args.attachment_sha256, args.batch)
        print(f"[time] {name} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
