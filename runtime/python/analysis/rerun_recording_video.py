"""Rerun a recording through a selected model and render its final live pose."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.compare_video import render
from production.deadbug_replay import replay
from production.paths import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--after", type=float, default=None,
                    help="render only outputs represented at or after this recording time")
    ap.add_argument("--initial-calibration-only", action="store_true")
    ap.add_argument("--no-waist-constraint", action="store_true")
    ap.add_argument("--v3-pelvis", action="store_true",
                    help="use the historical analysis/lyingcat_variants/v3_pelvis rule")
    args = ap.parse_args()
    recording = Path(args.recording)
    if not recording.exists():
        recording = paths.recordings / args.recording
    if args.v3_pelvis:
        import production.frame_chain as frame_chain
        from analysis.lyingcat_variants.v3_pelvis import LyingCatContact
        frame_chain.LyingCatContact = LyingCatContact
    elif args.no_waist_constraint:
        import production.frame_chain as frame_chain
        from analysis.lyingcat_variants.v5_no_waist import LyingCatContact
        frame_chain.LyingCatContact = LyingCatContact
    result = replay(recording, model_id=args.model, start_context_frames=60, start_offset_frames=60,
                    initial_calibration_only=args.initial_calibration_only)
    local = result.final_local
    times = result.represented_frames.astype(float) / 60.0
    if args.after is not None:
        keep = times >= args.after
        local, times = local[keep], times[keep]
    key = "rerun"
    text = {key: [[f"model={args.model}"] for _ in range(len(local))]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out, [(key, "MobilePose AMASS 重跑输出", local)], times, text,
           f"{recording.name} | rerun model={args.model}"
           + (" | no waist constraint" if args.no_waist_constraint else ""), frame_key=key,
           workers=args.workers)
    print(out, flush=True)


if __name__ == "__main__":
    main()
