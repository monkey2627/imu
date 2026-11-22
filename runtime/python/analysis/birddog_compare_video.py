"""Side-by-side SMPL mesh video of the bird-dog support-knee rule (production/birddog_contact.py).

Columns: ① current live chain with the bird-dog rule switched off | ② = production (current live chain,
rule on).  Both are production.live_replay.replay_live(simulate=True): the current run.py logic
(start-pose fit at Unity's birdDogBaseCalibrate, rebase, rules) on the recorded input.
Top row: side view; bottom row: 3/4 view; each body's lowest vertex on the floor.
Per-frame text: thigh elevation (deg, -90 = straight down) and knee angle (deg, 180 = straight) per
leg; column ② also the rule state (K = kneeling weight, UP = lift detected).

    python -m analysis.birddog_compare_video [recording ...] [--workers 8]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from analysis.compare_video import render
from production.arm_prior import _globals
from production.paths import paths

SUBTITLES = {
    "birddog_20261123_021217": "起始四点跪姿 0-13s；对侧伸展 14-17s(左手右腿) 20-23s(右手左腿)；只伸手 26/30/34/37s；"
                               "只伸腿 40-42/45-47/49-52/54-56s；60s 起身",
    "birddog_20261123_005257": "起始四点跪姿 0-15s（原问题：左腿越抬越高）；之后对侧伸展 6 次，43-44s 右腿伸至水平",
}
DOWN = np.array([0.0, -1.0, 0.0])


def _deg(a, b):
    return math.degrees(math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def leg_text(local):
    g = _globals(np.asarray(local, np.float64))
    parts = []
    for name, thigh, shank in (("L", 1, 4), ("R", 2, 5)):
        t, s = g[thigh] @ DOWN, g[shank] @ DOWN
        elev = math.degrees(math.asin(float(np.clip(t[1] / np.linalg.norm(t), -1, 1))))
        parts.append(f"{name}: thigh {elev:4.0f}  knee {180 - _deg(t, s):4.0f}")
    return parts


def replay(path, rule_on):
    import production.frame_chain as frame_chain
    import production.live_replay as live_replay
    diags = []
    original = live_replay.postprocess_output

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        diags.append(result[2].get("birddog"))
        return result

    saved = frame_chain.BIRDDOG_ACTIONS
    live_replay.postprocess_output = wrapped
    if not rule_on:
        frame_chain.BIRDDOG_ACTIONS = frozenset()
    try:
        result = live_replay.replay_live(path, simulate=True)
    finally:
        frame_chain.BIRDDOG_ACTIONS = saved
        live_replay.postprocess_output = original
    # live_replay postprocesses every output twice (replayed chain, then the rules-only check): keep the first
    return result, diags[0::2][:len(result.times)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*",
                        default=["birddog_20261123_021217.jsonl", "birddog_20261123_005257.jsonl"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    for name in args.recordings:
        path = paths.recordings / name
        off, _ = replay(path, rule_on=False)
        on, diags = replay(path, rule_on=True)
        columns = [("off", "① 关闭鸟狗规则（当前链路）", off.final_local.astype(np.float64)),
                   ("on", "② 当前生产：鸟狗支撑膝规则", on.final_local.astype(np.float64))]
        text = {"off": [leg_text(local) for local in columns[0][2]], "on": []}
        for local, diag in zip(columns[1][2], diags):
            lines = leg_text(local)
            if diag:
                w, up = diag["weights"], diag["lifted"]
                lines.append(f"K L {w['left']:.2f}{' UP' if up['left'] else ''}   "
                             f"K R {w['right']:.2f}{' UP' if up['right'] else ''}")
            text["on"].append(lines)
        stem = Path(name).stem
        out = paths.recordings / f"birddog_compare_{stem}.mp4"
        render(out, columns, on.times, text, SUBTITLES.get(stem, stem), workers=args.workers)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
