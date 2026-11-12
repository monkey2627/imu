#!/usr/bin/env python3
"""Sanity-check SlimeVR mounting/reset using a short `run --record` of you standing still.

    python -m production.run --model pnp5_60a_s42_ftr3 --record recordings/stand.jsonl   # stand upright ~5 s, Ctrl+C
    python scripts/check_standing_orientation.py recordings/stand.jsonl [--arms {down,tpose}]

The model input orientation of each sensor is the global rotation of its bone in the SMPL frame
(SMPL T-pose = identity).  Standing upright therefore gives

  pelvis, left/right shank : ~0 deg from identity
  forearms, arms down      : ~90 deg (a +-90 deg rotation about Z), arms in T-pose: ~0 deg

A value far from these (e.g. ~90 or ~180 deg for the pelvis/shanks) means the tracker's mounting
orientation, the full reset or the body-part binding is wrong.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

LABELS = ["lw", "rw", "ls", "rs", "pelvis"]


def angle_to_identity(rotations):
    cos = np.clip((np.trace(rotations, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", type=Path)
    p.add_argument("--arms", choices=("down", "tpose"), default="down", help="arm pose while recording (default: down at the sides)")
    p.add_argument("--seconds", type=float, default=3.0, help="use the last N seconds of the recording")
    p.add_argument("--tolerance", type=float, default=20.0, help="allowed deviation in degrees")
    args = p.parse_args(argv)

    rows = [json.loads(line) for line in args.recording.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = rows[0]
    fps = float(meta.get("modelFps", 60))
    feats = [np.asarray(r["feature"], dtype=np.float64) for r in rows if r.get("type") in ("windowFrame", "inferenceFrame") and "feature" in r]
    if not feats:
        print("no feature rows in the recording (record with the current run.py)", file=sys.stderr)
        return 2
    feats = np.stack(feats)[-int(args.seconds * fps):]
    angles = angle_to_identity(feats[:, 15:60].reshape(-1, 5, 3, 3))          # (frames, 5)
    median = np.median(angles, axis=0)
    spread = np.percentile(angles, 90, axis=0) - np.percentile(angles, 10, axis=0)
    expected = {"lw": 90.0 if args.arms == "down" else 0.0, "rw": 90.0 if args.arms == "down" else 0.0,
                "ls": 0.0, "rs": 0.0, "pelvis": 0.0}
    bad = False
    print(f"{len(feats)} frames ({len(feats) / fps:.1f} s); angle of each sensor's bone rotation from SMPL T-pose identity")
    for k, label in enumerate(LABELS):
        ok = abs(median[k] - expected[label]) <= args.tolerance
        bad |= not ok
        print(f"  {label:7s} {median[k]:6.1f} deg  (expected ~{expected[label]:.0f}, spread {spread[k]:.1f})  {'OK' if ok else 'CHECK mounting / reset / binding'}")
    if spread.max() > 10:
        print("  note: large spread -> you moved during the recording; stand still and repeat")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
