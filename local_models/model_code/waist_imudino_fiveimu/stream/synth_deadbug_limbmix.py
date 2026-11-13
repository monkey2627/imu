"""Synthesize DeadBug and train-only upper/lower-body recombinations at 60 Hz."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from downstream.data import group_fold
from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.no_head_layouts import generate_layout_dataset, _synthesize, _foot_ground_probs, _save_dataset

LEG = [1, 2, 4, 5, 7, 8, 10, 11]
ARM = [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def rest_frame(pose, kind):
    """Dead-bug neutral: thighs/upper arms point up, knees bent, arms extended.

    World-up scoring works for supine motion regardless of camera heading.
    Forearms alone cannot distinguish a lowered upper arm with a bent elbow.
    """
    global_rot = []
    for j, parent in enumerate(PARENTS):
        global_rot.append(pose[:, j] if parent < 0 else global_rot[parent] @ pose[:, j])
    specs = [(1, (0, -1, 0)), (2, (0, -1, 0))] if kind == "legs" else [
        (16, (1, 0, 0)), (17, (-1, 0, 0)),
        (18, (1, 0, 0)), (19, (-1, 0, 0))]
    up = pose.new_tensor([0, 1, 0])
    angles = [torch.rad2deg(torch.acos(((global_rot[j] @ pose.new_tensor(axis)) @ up).clamp(-1, 1)))
              for j, axis in specs]
    score = torch.stack(angles).mean(0)
    if kind == "legs":
        # Prefer tabletop knees (90 degrees), without requiring an exact angle.
        bends = torch.stack([torch.rad2deg(torch.acos(
            ((pose[:, j].diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
            for j in (4, 5)])
        score = score + 0.25 * (bends - 90).abs().mean(0)
    frame = int(score.argmin())
    return frame, float(torch.stack(angles)[:, frame].mean())


def bind_groups(source, raw_root):
    files = sorted(raw_root.glob("*/*_poses.npz"))
    if len(files) != len(source["pose"]):
        raise ValueError("Raw/processed sequence count mismatch")
    groups = []
    for file, pose in zip(files, source["pose"]):
        meta = json.loads(file.with_name(file.name.replace("_poses.npz", ".json")).read_text())
        with np.load(file) as clip:
            if float(clip["mocap_framerate"]) != 60 or len(clip["poses"]) != len(pose):
                raise ValueError(f"Raw/processed frame mismatch: {file}")
        if meta["num_frames"] != len(pose):
            raise ValueError(f"Sidecar frame mismatch: {file}")
        groups.append(f"DeadBug:{meta['source_video_sha256']}")
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    # Redirect this process only; the shared synthesis workspace is unchanged.
    paths.raw_amass = args.raw.parent
    if args.raw.name != "DeadBug":
        raise ValueError("Expected the DeadBug dataset directory")
    generate_layout_dataset("wrists_shanks_waist", args.out, mode="surface",
                            attachment_config=args.config, mesh_chunk_size=128,
                            dataset_names=["DeadBug"])
    source_path = args.out / "DeadBug.pt"
    src = torch.load(source_path, map_location="cpu", weights_only=False)
    groups = bind_groups(src, args.raw)
    src["groups"] = groups
    _save_dataset(src, source_path)
    train = [i for i, group in enumerate(groups) if group_fold(group) == "train" and len(src["pose"][i]) >= 64]
    heldout = {g for g in groups if group_fold(g) == "val"}
    if not heldout or len({groups[i] for i in train}) < 2:
        raise ValueError("Need held-out videos and at least two training videos")
    model = ParametricModel(paths.smpl_file)
    out = {k: [] for k in ("pose", "shape", "tran", "joint", "acc", "ori", "contact")}
    meta = []
    skipped = {"armonly": 0, "legonly": 0}
    for i in train:
        for kind in ("cross", "armonly", "legonly"):
            pose = src["pose"][i].clone()
            partner, rest = "", None
            if kind == "cross":
                j = random.choice([x for x in train if groups[x] != groups[i]])
                other = src["pose"][j]
                n = min(len(pose), len(other))
                offset = random.randint(0, len(other) - n)
                pose = pose[:n].clone()
                pose[:, LEG] = other[offset:offset + n][:, LEG]
                partner = groups[j]
            else:
                rest, deg = rest_frame(pose, "legs" if kind == "armonly" else "arms")
                if deg > 35:
                    skipped[kind] += 1
                    continue
                joints = LEG if kind == "armonly" else ARM
                pose[:, joints] = pose[rest, joints]
            tran = src["tran"][i][:len(pose)].clone()
            ori, joint, acc = _synthesize(model, pose, src["shape"][i], tran,
                                         "wrists_shanks_waist", src["synthesis"], 128)
            values = dict(pose=pose, shape=src["shape"][i].clone(), tran=tran,
                          joint=joint[:, :24].contiguous(), acc=acc, ori=ori,
                          contact=_foot_ground_probs(joint))
            for key, value in values.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite {kind} {groups[i]} {key}")
                out[key].append(value)
            meta.append(dict(kind=kind, source=groups[i], partner=partner, rest_frame=rest))
    counts = {kind: sum(m["kind"] == kind for m in meta) for kind in ("cross", "armonly", "legonly")}
    if not all(counts.values()):
        raise ValueError(f"Missing augmentation kind: {counts}")
    assert all(m["source"] not in heldout and m["partner"] not in heldout for m in meta)
    data = {key: src[key] for key in ("synthesis", "layout", "layout_labels", "layout_joints")}
    data.update(out, groups=[f"DeadBugMix:{m['kind']}:{m['source']}|{m['partner']}" for m in meta],
                train_only=True, mix_meta=meta)
    _save_dataset(data, args.out / "DeadBugMix.pt")
    report = dict(train_source_clips=len(train), val_source_clips=sum(group_fold(g) == "val" for g in groups),
                  source_videos=len(set(groups)), val_videos=len(heldout), counts=counts, skipped=skipped,
                  mix_frames=sum(map(len, out["pose"])), seed=args.seed)
    (args.out / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
