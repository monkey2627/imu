"""Decouple arm reach from leg lift in BirdDog: recombine limbs, then re-synthesize IMU.

BirdDog pseudo-labels almost always pair one arm reach with the opposite leg lift,
so a model learns "arm forward => opposite thigh up" even when the shanks are
static (recording 20261112_015543). This builds BirdDogMix from TRAIN-fold clips
only, with three equally likely kinds per output clip:

* cross:   trunk/arms (root, spine, neck, head, arms) from clip A, legs from clip B;
* armonly: clip A, legs frozen at A's own kneeling frame (both thighs near vertical);
* legonly: clip A, arms frozen at A's own rest frame (both forearms near vertical).

Root orientation/translation always come from A. IMU is synthesized with the same
surface attachments and code as every other dataset (no_head_layouts._synthesize).
All sequences are marked train-only; validation stays on untouched BirdDog groups.

Run inside the synthesis workspace (PYTHONPATH=<synth_ws>, IMUDINO repo on PYTHONPATH for group_fold):
    python synth_birddog_limbmix.py --birddog <processed>/BirdDog.pt --raw <flowhmr BirdDog root> \
        --config <ws>/mobileposer/configs/wristankle_surface_imu.json --out <processed>/BirdDogMix.pt
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.no_head_layouts import LAYOUTS, _foot_ground_probs, _save_dataset, _synthesize, synthesis_metadata

LEG = [1, 2, 4, 5, 7, 8, 10, 11]
ARM = [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
DOWN = torch.tensor([0.0, -1.0, 0.0])


def global_rot(local):
    g = [None] * 24
    for j, p in enumerate(PARENTS):
        g[j] = local[:, j] if p < 0 else g[p] @ local[:, j]
    return torch.stack(g, 1)


def bone_pitch(g, joint, axis):
    v = g[:, joint] @ torch.tensor(axis, dtype=g.dtype)
    return torch.rad2deg(torch.arccos((v @ DOWN.to(g.dtype)).clamp(-1, 1)))


def rest_frame(pose, kind):
    """Frame index where both thighs (kind=legs) or both forearms (kind=arms) hang most vertically."""
    g = global_rot(pose)
    if kind == "legs":
        score = bone_pitch(g, 1, [0, -1, 0]) + bone_pitch(g, 2, [0, -1, 0])
    else:
        score = bone_pitch(g, 18, [1, 0, 0]) + bone_pitch(g, 19, [-1, 0, 0])
    return int(score.argmin()), float(score.min() / 2)


def bird_dog_groups(processed_path, raw_root):
    """Same mapping as downstream.prepare.bird_dog_groups (inlined to avoid importing the backbone)."""
    import json
    processed = torch.load(processed_path, map_location="cpu", weights_only=False)
    raw_files = sorted(Path(raw_root).glob("*/*_poses.npz"))
    if len(raw_files) != len(processed["pose"]):
        raise ValueError("BirdDog raw/processed sequence counts differ")
    groups = []
    for raw_file, pose in zip(raw_files, processed["pose"]):
        meta = json.loads(raw_file.with_name(raw_file.name.replace("_poses.npz", ".json")).read_text(encoding="utf-8"))
        if meta["num_frames"] != len(pose):
            raise ValueError(f"BirdDog frame alignment failed: {raw_file}")
        groups.append(f"BirdDog:{meta['source_video_sha256']}")
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--birddog", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copies", type=int, default=3, help="output clips per train clip")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    from downstream.data import group_fold  # torch-only module of the IMUDINO repo

    random.seed(args.seed)
    src = torch.load(args.birddog, map_location="cpu", weights_only=False)
    synthesis = synthesis_metadata("surface", args.config)
    if src["synthesis"]["attachment_sha256"] != synthesis["attachment_sha256"]:
        raise ValueError("BirdDog source was synthesized with another attachment config")
    groups = bird_dog_groups(args.birddog, args.raw)
    train = [i for i, g in enumerate(groups) if group_fold(g) == "train" and len(src["pose"][i]) >= 64]
    model = ParametricModel(paths.smpl_file)
    out = {k: [] for k in ("pose", "shape", "tran", "joint", "acc", "ori", "contact")}
    meta, kinds = [], {"cross": 0, "armonly": 0, "legonly": 0}
    for i in train:
        for c in range(args.copies):
            kind = ("cross", "armonly", "legonly")[c % 3]
            a = src["pose"][i].clone()
            if kind == "cross":
                j = random.choice([x for x in train if groups[x] != groups[i]])
                b = src["pose"][j]
                n = min(len(a), len(b))
                off = random.randint(0, len(b) - n)
                a, tran = a[:n].clone(), src["tran"][i][:n]
                a[:, LEG] = b[off:off + n][:, LEG]
                partner = groups[j]
            else:
                tran = src["tran"][i]
                k, deg = rest_frame(a, "legs" if kind == "armonly" else "arms")
                if deg > 35:            # this clip never really rests that limb pair; skip
                    continue
                a[:, LEG if kind == "armonly" else ARM] = a[k, LEG if kind == "armonly" else ARM]
                partner = ""
            ori, joint, acc = _synthesize(model, a, src["shape"][i], tran, "wrists_shanks_waist", synthesis, 128)
            out["pose"].append(a); out["shape"].append(src["shape"][i].clone()); out["tran"].append(tran.clone())
            out["joint"].append(joint[:, :24].contiguous()); out["acc"].append(acc); out["ori"].append(ori)
            out["contact"].append(_foot_ground_probs(joint))
            meta.append({"kind": kind, "source": groups[i], "partner": partner})
            kinds[kind] += 1
    layout = LAYOUTS["wrists_shanks_waist"]
    data = {"synthesis": synthesis, "layout": "wrists_shanks_waist", "layout_labels": layout["labels"],
            "layout_joints": layout["joints"], **out,
            "groups": [f"BirdDogMix:{m['kind']}:{m['source']}|{m['partner']}" for m in meta],
            "train_only": True, "mix_meta": meta}
    _save_dataset(data, args.out)
    print({"train_source_clips": len(train), "output_clips": len(meta), **kinds,
           "frames": int(sum(len(p) for p in out["pose"]))})


if __name__ == "__main__":
    main()
