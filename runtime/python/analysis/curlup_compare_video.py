"""Curl-up comparison video on the real recording: network vs production chain vs bridge model.

    python -m analysis.curlup_compare_video [recording ...] [--workers 8]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import production.mesh_video as mv
from analysis.compare_video import angle_between, fk, render, smpl_model
from production.deadbug_replay import replay
from production.lyingcat_contact import LyingCatContact
from production.model_backends import ACTION_RUNTIME_MODELS
from production.paths import paths
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw

DEFAULT_RECORDINGS = ("unity_20261116_023352.jsonl",)
BRIDGE_MODEL = ACTION_RUNTIME_MODELS["hipthrust"]
DOWN = np.array([0.0, -1.0, 0.0])
BELOW = {"head": (12, 15), "shoulder": (13, 14, 16, 17), "foot": (7, 8, 10, 11), "hand": (18, 19, 20, 21, 22, 23)}


def _text(local, offsets, model, dominant):
    g, _ = fk(local, offsets)
    up = -(g[9] @ DOWN)                       # spine3 axis pointing to the neck
    trunk = float(np.degrees(np.arcsin(np.clip(up[1], -1, 1))))
    knee = [angle_between(g[h] @ DOWN, g[k] @ DOWN) for h, k in ((1, 4), (2, 5))]
    hip = [angle_between(g[0] @ DOWN, g[h] @ DOWN) for h in (1, 2)]
    verts = mv.skin(model, batch_smpl_matrix_to_unity_quat_xyzw(np.asarray(local, np.float32)), False)
    pelvis = verts[np.isin(dominant, (0,)), 1].min()
    below = []
    for name, bones in BELOW.items():
        depth = 100.0 * (verts[np.isin(dominant, bones), 1].min() - pelvis)
        if depth < -2.0:
            below.append(f"{name} {depth:.0f}")
    return (f"upper trunk lift {trunk:4.0f} deg",
            f"knee L {knee[0]:3.0f} R {knee[1]:3.0f}  hip L {hip[0]:3.0f} R {hip[1]:3.0f}",
            "below pelvis (cm): " + (", ".join(below) if below else "none"))


def build(recording, workers):
    model_id = ACTION_RUNTIME_MODELS["crunch"]
    common = dict(start_context_frames=60, start_offset_frames=60)
    path = paths.recordings / recording
    current = replay(path, model_id=model_id, **common)
    bridge = replay(path, model_id=BRIDGE_MODEL, **common)
    columns = [
        ("network", "① 卷腹模型 网络原始输出", current.raw_local),
        ("current", "② 当前生产：卷腹模型＋卷腹规则", current.final_local),
        ("bridge", "③ 参考：臀桥模型＋臀桥规则（录制时误选）", bridge.final_local),
    ]
    n = min(len(c[2]) for c in columns)
    columns = [(k, t, l[:n]) for k, t, l in columns]
    offsets = LyingCatContact().offsets
    model, _faces = smpl_model()
    dominant = model["bone_ids"][np.arange(len(model["bone_ids"])), model["weights"].argmax(1)]
    text = {key: [_text(pose, offsets, model, dominant) for pose in local] for key, _t, local in columns}
    times = current.represented_frames[:n] / 60.0
    subtitle = (f"{recording} · 生产模型 {model_id} · 上排侧视 / 下排 3/4 · 最低点落地 · "
                "青色：上躯干抬起角、膝屈/髋屈、低于骨盆的部位")
    out = paths.recordings / "action_eval_20261119" / f"curlup_compare_{Path(recording).stem}.mp4"
    return render(out, columns, times, text, subtitle, frame_key="current", workers=workers), text, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*", default=list(DEFAULT_RECORDINGS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    for recording in args.recordings:
        print(recording, flush=True)
        out, _text_, _t = build(recording, args.workers)
        print(out, flush=True)


if __name__ == "__main__":
    main()
