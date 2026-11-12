"""Compare Unity input recordings of the same action across wearings.

Answers "is the difference between wearings bigger than the difference between
two repeats with the same wearing?" on three levels:

1. mount axis   -- the dominant rotation axis of each sensor while moving,
                   expressed in its own (bone) frame. A mounting error of
                   theta about the bone tilts this axis by ~theta. A SlimeVR
                   reset makes the reset pose look identical for any mounting,
                   so only motion reveals it.
2. input        -- DTW-aligned angle between waist-relative sensor orientations.
3. output       -- DTW-aligned angle between model joint rotations per joint
                   group, for each requested model (replayed offline exactly
                   like the live loop: prime with the recorded upright pose,
                   infer every 2 frames).

Usage (from realUse/):
    python -m production.compare_recordings A1=same_1.jsonl A2=same_2.jsonl \
        B=rewear_reset.jsonl C=rewear_noreset.jsonl --report recordings/compare.json
Names are optional; files resolve like --replay (bare names under realUse/recordings).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from production.model_backends import make_waist_session, resolve_waist_model
from production.calibration import bone_orientation, identity_heading
from production.input_recording import load_input_recording, resolve_recording_path
from production.layout import ResolvedLayout

DEFAULT_MODELS = ("waist_mobileposer_ours_all10", "waist_streamposer_ours_all10", "waist_line14_ours_all10")
JOINT_GROUPS = {
    "hips": [1, 2], "knees": [4, 5], "spine": [3, 6, 9],
    "upper_arms": [16, 17], "elbows": [18, 19],
}
ANALYSIS_HZ = 10           # DTW resolution
MOTION_THRESHOLD_DEG = 15  # samples closer than this to the upright pose do not define a mount axis


def _frames(recording):
    """Expand recorded rows to the 60 Hz model timeline (rows advanced by `ticks`)."""
    key = "reference_adjusted_accel_xyz" if recording.acceleration_source == "server" else "accel_xyz"
    quats, accels = [], []
    for row, ticks in zip(recording.samples, recording.ticks):
        quats += [[row[label]["quat_xyzw"] for label in recording.labels]] * ticks
        accels += [[row[label][key] for label in recording.labels]] * ticks
    return np.asarray(quats, dtype=np.float32), np.asarray(accels, dtype=np.float32)


def _orientations(recording, quats):
    return np.stack([bone_orientation(q, recording.labels).numpy() for q in quats])   # (T, S, 3, 3)


def _angle(a, b):
    """Geodesic angle in degrees between stacks of rotation matrices."""
    trace = np.einsum("...ji,...ji->...", a, b)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))


def _mount_axes(recording, ori):
    """Dominant body-frame rotation axis per sensor relative to the upright pose."""
    if recording.upright is not None:
        reference = bone_orientation(recording.upright[0], recording.labels).numpy()
    else:
        reference = ori[0]
    axes = {}
    for s, label in enumerate(recording.labels):
        delta = np.swapaxes(reference[s], -1, -2)[None] @ ori[:, s]
        rotvec = Rotation.from_matrix(delta).as_rotvec()
        moving = rotvec[np.linalg.norm(rotvec, axis=1) > np.radians(MOTION_THRESHOLD_DEG)]
        if len(moving) < 10:
            axes[label] = None
            continue
        _, _, vt = np.linalg.svd(moving, full_matrices=False)
        axes[label] = vt[0]
    return axes


def _waist_relative(recording, ori):
    waist = recording.labels.index("waist")
    rel = np.swapaxes(ori[:, waist:waist + 1], -1, -2) @ ori
    rel[:, waist] = ori[:, waist]
    return rel


def _dtw_path(a, b):
    """Plain DTW on (T, D) feature sequences; returns matched index pairs."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None] - b[None], axis=-1)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row, prev = acc[i], acc[i - 1]
        for j in range(1, m + 1):
            row[j] = cost[i - 1, j - 1] + min(prev[j], prev[j - 1], row[j - 1])
    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin((acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]))
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return np.asarray(path[::-1])


def _run_model(model_id, recording, quats, accels):
    resolved = ResolvedLayout(recording.layout, recording.labels, dict(recording.label_to_key))
    heading = identity_heading(recording.labels)
    session = make_waist_session(model_id, resolved, heading, loaded=resolve_waist_model(model_id))
    if recording.upright is not None:
        session.prime(*recording.upright)
    poses, last = [], None
    for index, (q, a) in enumerate(zip(quats, accels)):
        feature, ori, acc = session.prepare_feature(q, a)
        if hasattr(session, "push_observation"):
            session.push_observation(feature, ori, acc)
        else:
            session.push_feature(feature)
        if index % 2 == 1 and index >= 11:   # live cadence; Line14 needs >=10 frames
            last = session.infer().detach().numpy()
        poses.append(last if last is not None else np.tile(np.eye(3, dtype=np.float32), (24, 1, 1)))
    return np.asarray(poses)   # (T, 24, 3, 3) local rotations at 60 Hz


def analyse(named_paths, models):
    data = {}
    for name, path in named_paths:
        recording = load_input_recording(path)
        quats, accels = _frames(recording)
        ori = _orientations(recording, quats)
        data[name] = {"recording": recording, "quats": quats, "accels": accels, "ori": ori,
                      "rel": _waist_relative(recording, ori), "axes": _mount_axes(recording, ori)}
        print(f"loaded {name}: {path.name} ({recording.duration:.1f}s)", file=sys.stderr)
    labels = next(iter(data.values()))["recording"].labels
    step = max(1, round(60 / ANALYSIS_HZ))

    for model_id in models:
        for name, item in data.items():
            print(f"running {model_id} on {name}...", file=sys.stderr)
            item.setdefault("poses", {})[model_id] = _run_model(model_id, item["recording"], item["quats"], item["accels"])

    pairs = []
    for (na, a), (nb, b) in itertools.combinations(data.items(), 2):
        fa, fb = a["rel"][::step], b["rel"][::step]
        path = _dtw_path(fa.reshape(len(fa), -1), fb.reshape(len(fb), -1))
        ia, ib = path[:, 0] * step, path[:, 1] * step
        mount = {}
        for label in labels:
            ua, ub = a["axes"][label], b["axes"][label]
            mount[label] = None if ua is None or ub is None else float(np.degrees(np.arccos(min(1.0, abs(ua @ ub)))))
        sensor = {label: float(np.mean(_angle(a["rel"][ia, s], b["rel"][ib, s]))) for s, label in enumerate(labels)}
        output = {}
        for model_id in models:
            pa, pb = a["poses"][model_id][ia], b["poses"][model_id][ib]
            output[model_id] = {group: float(np.mean(_angle(pa[:, joints], pb[:, joints])))
                                for group, joints in JOINT_GROUPS.items()}
        pairs.append({"a": na, "b": nb, "mountAxisDeg": mount, "inputDeg": sensor, "outputDeg": output})
    return {"recordings": {name: str(item["recording"].path) for name, item in data.items()},
            "models": list(models), "pairs": pairs}


def _fmt(value):
    return "  -  " if value is None else f"{value:5.1f}"


def print_report(report):
    pairs = report["pairs"]
    labels = list(pairs[0]["inputDeg"])
    print("\n[1] Mount-axis offset per sensor (deg; repeat-vs-repeat ~ noise floor, rewear larger = mounting changed)")
    print("pair            " + " ".join(f"{label:>6}" for label in labels))
    for p in pairs:
        print(f"{p['a'] + ' vs ' + p['b']:<16}" + " ".join(f"{_fmt(p['mountAxisDeg'][l]):>6}" for l in labels))
    print("\n[2] DTW-aligned input difference (deg; limbs relative to waist, waist absolute)")
    print("pair            " + " ".join(f"{label:>6}" for label in labels))
    for p in pairs:
        print(f"{p['a'] + ' vs ' + p['b']:<16}" + " ".join(f"{_fmt(p['inputDeg'][l]):>6}" for l in labels))
    print("\n[3] DTW-aligned model output difference (deg, local joint rotation)")
    groups = list(JOINT_GROUPS)
    for model_id in report["models"]:
        print(f"  {model_id}")
        print("  pair            " + " ".join(f"{g:>10}" for g in groups))
        for p in pairs:
            print(f"  {p['a'] + ' vs ' + p['b']:<16}" + " ".join(f"{p['outputDeg'][model_id][g]:10.1f}" for g in groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+", help="NAME=file.jsonl or file.jsonl (at least two)")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated model ids; empty = inputs only")
    parser.add_argument("--report", type=Path, help="write the full JSON report here")
    args = parser.parse_args()
    if len(args.recordings) < 2:
        parser.error("need at least two recordings")
    named = []
    for index, item in enumerate(args.recordings):
        name, _, value = item.partition("=") if "=" in item else (f"R{index + 1}", "", item)
        named.append((name, resolve_recording_path(value)))
    models = [m for m in args.models.split(",") if m]
    torch.set_grad_enabled(False)
    report = analyse(named, models)
    print_report(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport: {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
