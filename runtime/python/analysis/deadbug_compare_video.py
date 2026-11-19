"""Dead-bug comparison video on real recordings: network vs previous chain vs back-on-floor contact.

    python -m analysis.deadbug_compare_video [recording ...] [--workers 8]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import production.frame_chain as frame_chain
import production.mesh_video as mv
from analysis.compare_video import angle_between, fk, render, smpl_model
from production.deadbug_floor import DeadBugFloorContact
from production.deadbug_replay import replay
from production.lyingcat_contact import LyingCatContact
from production.model_backends import ACTION_RUNTIME_MODELS
from production.paths import paths
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw

DEFAULT_RECORDINGS = ("unity_20261116_045157.jsonl", "unity_20261116_063451.jsonl")
DOWN = np.array([0.0, -1.0, 0.0])
BELOW = {"head": (12, 15), "heel": (7, 8, 10, 11), "knee": (1, 2, 4, 5)}


class _NoFloor:
    def __init__(self):
        pass

    def reset(self, reason):
        pass

    def apply(self, local, dt):
        return local, {}


def _text(local, offsets, model, dominant):
    g, _ = fk(local, offsets)
    trunk = g[3] @ DOWN
    knee = [angle_between(g[h] @ DOWN, g[k] @ DOWN) for h, k in ((1, 4), (2, 5))]
    hip = [angle_between(trunk, g[h] @ DOWN) for h in (1, 2)]
    verts = mv.skin(model, batch_smpl_matrix_to_unity_quat_xyzw(np.asarray(local, np.float32)), False)
    back = verts[np.isin(dominant, (0, 3, 6, 9)), 1].min()
    below = []
    for name, bones in BELOW.items():
        depth = 100.0 * (verts[np.isin(dominant, bones), 1].min() - back)
        if depth < -1.0:
            below.append(f"{name} {depth:.0f}")
    return (f"knee flex L {knee[0]:3.0f} R {knee[1]:3.0f} deg",
            f"hip flex  L {hip[0]:3.0f} R {hip[1]:3.0f} deg",
            "below back (cm): " + (", ".join(below) if below else "none"))


def build(recording, workers):
    model_id = ACTION_RUNTIME_MODELS["deadbug"]
    common = dict(model_id=model_id, start_context_frames=60, start_offset_frames=60,
                  start_pose_override=("deadbug", "tabletop"))
    path = paths.recordings / recording
    frame_chain.DeadBugFloorContact = _NoFloor
    previous = replay(path, **common)
    frame_chain.DeadBugFloorContact = DeadBugFloorContact
    current = replay(path, **common)
    columns = [
        ("network", "① 网络原始输出", current.raw_local),
        ("previous", "② 上一版：关 deadbug_legs，无地面约束", previous.final_local),
        ("current", "③ 当前：②＋背部贴地（腿/头不低于背）", current.final_local),
    ]
    n = min(len(c[2]) for c in columns)
    columns = [(k, t, l[:n]) for k, t, l in columns]
    offsets = LyingCatContact().offsets
    model, _faces = smpl_model()
    dominant = model["bone_ids"][np.arange(len(model["bone_ids"])), model["weights"].argmax(1)]
    text = {key: [_text(pose, offsets, model, dominant) for pose in local] for key, _t, local in columns}
    times = current.represented_frames[:n] / 60.0
    subtitle = (f"{recording} · 模型 {model_id} · 起始姿态 tabletop 约束开启 · 上排侧视 / 下排 3/4 · 最低点落地 · "
                "青色：膝屈、髋屈、低于背部的部位（none = 背部是最低点）")
    out = paths.recordings / "action_eval_20261119" / f"deadbug_compare_{Path(recording).stem}.mp4"
    return render(out, columns, times, text, subtitle, frame_key="current", workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*", default=list(DEFAULT_RECORDINGS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    for recording in args.recordings:
        print(recording, flush=True)
        print(build(recording, args.workers), flush=True)


if __name__ == "__main__":
    main()
