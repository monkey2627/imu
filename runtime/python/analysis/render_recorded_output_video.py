"""Render the already-recorded realtime output without running model inference.

The recording's ``prediction.unityLocalQuaternions`` are the final poses that
were broadcast by the live pipeline.  This utility only converts those poses
to SMPL rotations and renders them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.compare_video import render
from production.live_replay import _unity_quats_to_smpl
from production.paths import paths


def build(recording: str, out: str | None, workers: int = 8):
    path = Path(recording)
    if not path.exists():
        path = paths.recordings / recording
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = [row for row in rows if row.get("type") == "prediction"]
    if not predictions:
        raise RuntimeError(f"no prediction rows in {path}")
    local = np.stack([_unity_quats_to_smpl(row["unityLocalQuaternions"]) for row in predictions])
    times = np.asarray([float(row.get("t", i / 30.0)) for i, row in enumerate(predictions)], dtype=float)
    model = predictions[0].get("model", "recorded realtime output")
    action = predictions[0].get("actionId") or "unknown"
    key = "recorded"
    text = {key: [[f"action={action}", f"model={model}"] for _ in predictions]}
    subtitle = f"{path.name} | recorded unityLocalQuaternions | no inference"
    if out is None:
        out_path = path.parent / "action_eval_20261119" / f"recorded_live_{path.stem}.mp4"
    else:
        out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return render(out_path, [(key, "实时链路已记录输出", local)], times, text, subtitle,
                  frame_key=key, workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--out")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(build(args.recording, args.out, args.workers), flush=True)


if __name__ == "__main__":
    main()
