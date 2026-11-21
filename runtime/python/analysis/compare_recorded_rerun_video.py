"""Side-by-side recorded live output vs an offline model/rule rerun."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.compare_video import render
from production.deadbug_replay import replay
from production.live_replay import _unity_quats_to_smpl
from production.paths import paths


def build(recording, model, out, workers=1, after=None):
    path = Path(recording)
    if not path.exists():
        path = paths.recordings / recording
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preds = [row for row in rows if row.get("type") == "prediction"]
    live = np.stack([_unity_quats_to_smpl(row["unityLocalQuaternions"]) for row in preds])
    times_live = np.asarray([float(row["t"]) for row in preds])

    # Use the production contact rule, including the new hand/knee positional
    # anchors.  The historical v3 variant is intentionally not monkeypatched
    # here because this video validates the new realtime behavior.
    result = replay(path, model_id=model, start_context_frames=60, start_offset_frames=60)
    rerun = result.final_local
    times_rerun = result.represented_frames.astype(float) / 60.0

    n = min(len(live), len(rerun))
    live, rerun = live[:n], rerun[:n]
    times = times_rerun[:n]
    if after is not None:
        keep = times >= after
        live, rerun, times = live[keep], rerun[keep], times[keep]
    text = {
        "recorded": [["actual realtime output"] for _ in range(len(times))],
        "rerun": [[f"rerun: {model}", "v3_pelvis"] for _ in range(len(times))],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    return render(out, [("recorded", "录制时实时输出", live),
                        ("rerun", "离线 FlowHMR + v3 重跑", rerun)],
                  times, text, f"{path.name} | recorded vs rerun", frame_key="rerun", workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--model", default="waist_imudino_stream_lyingcat_flowhmr_20261118")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--after", type=float, default=8.83)
    ap.add_argument("--out-dir", default="runtime/python/recordings/action_eval_20261119")
    args = ap.parse_args()
    for recording in args.recordings:
        stem = Path(recording).stem
        out = Path(args.out_dir) / f"recorded_vs_rerun_{stem}.mp4"
        print(build(recording, args.model, out, args.workers, args.after), flush=True)


if __name__ == "__main__":
    main()
