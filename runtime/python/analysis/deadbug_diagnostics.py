"""Frame-aligned dead-bug diagnostics for Unity input recordings.

For each recording it runs the realtime replay chain twice (production.deadbug_replay):
an unconstrained baseline (dead-bug leg estimate off) and the current chain, then writes

* ``<stem>_deadbug_frames.csv``: one row per network output -- input frame, represented frame,
  lag, every sensor's reference-adjusted and calibrated orientation and angular speed, the raw and
  final hip/knee/shoulder/elbow angles per side, and the leg phase/confidence/reason;
* ``<stem>_deadbug_summary.json``: mapping checks, live-vs-offline reproduction error, angle and
  jump statistics, every "start extend"/"start return" decision with its evidence.

Usage (from runtime/python)::

    python -m analysis.deadbug_diagnostics unity_20261113_232942 unity_20261113_233607 unity_20261113_233649
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from production.arm_prior import PARENTS
from production.deadbug_replay import replay
from production.input_recording import load_input_recording, resolve_recording_path
from production.layout import BODY_PART_TO_LABEL
from production.model_backends import resolve_waist_model
from production.rotmat import UNITY_SMPL_BASIS

SIDES = (("L", 1, 4, 16, 18, "ls", "lw"), ("R", 2, 5, 17, 19, "rs", "rw"))
BONE_AXIS = {"lw": [1.0, 0, 0], "rw": [-1.0, 0, 0], "ls": [0, -1.0, 0], "rs": [0, -1.0, 0], "waist": [0, 1.0, 0]}
LABEL_JOINT = {"lw": 18, "rw": 19, "ls": 4, "rs": 5, "waist": 3}


def globals_of(local):
    g = np.empty_like(local)
    for joint, parent in enumerate(PARENTS):
        g[:, joint] = local[:, joint] if parent < 0 else g[:, parent] @ local[:, joint]
    return g


def angle_between(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def sagittal(v, waist):
    """Angle of v in the waist's sagittal plane: 0 = toward the feet, 90 = chest normal (up when supine)."""
    a, b = -waist[:, :, 1], waist[:, :, 2]
    return np.degrees(np.arctan2((v * b).sum(-1), (v * a).sum(-1)))


def limb_angles(local, waist):
    """Per side: thigh angle, knee flexion, upper-arm elevation toward the head, elbow flexion."""
    g = globals_of(local)
    down = np.array([0, -1.0, 0])
    out = {}
    for side, hip, knee, shoulder, elbow, _, _ in SIDES:
        thigh, shank = g[:, hip] @ down, g[:, knee] @ down
        axis = np.array([1.0, 0, 0]) if side == "L" else np.array([-1.0, 0, 0])
        upper, fore = g[:, shoulder] @ axis, g[:, elbow] @ axis
        out[side] = {
            "thigh": sagittal(thigh, waist),
            "knee": angle_between(thigh, shank),
            "shoulder": sagittal(upper, waist),     # 90 = arm toward the ceiling, 180 = overhead
            "elbow": angle_between(upper, fore),
        }
    return out


def step_jumps(values):
    if len(values) < 2:
        return np.zeros(1)
    return np.abs(np.diff(np.degrees(np.unwrap(np.radians(values)))))


def recorded_outputs(rows):
    """Live raw/final outputs with the input frame (count of tick rows before them) they were made at."""
    frames, raw, final = [], [], []
    ticks = 0
    for row in rows:
        if row.get("type") == "tick":
            ticks += int(row.get("ticks", 1))
        elif row.get("type") == "prediction" and isinstance(row.get("rawLocalQuaternions"), list):
            lag = int(row.get("diagnostics", {}).get("modelLagFrames", 1))
            frames.append(ticks - 1 - lag)
            to = lambda q: UNITY_SMPL_BASIS @ Rotation.from_quat(np.reshape(q, (24, 4))).as_matrix() @ UNITY_SMPL_BASIS
            raw.append(to(row["rawLocalQuaternions"]))
            final.append(to(row["unityLocalQuaternions"]))
    return np.asarray(frames), np.asarray(raw), np.asarray(final)


def mapping_check(path, rows, result):
    rec = load_input_recording(path)
    body_parts = {}
    for samples in rec.samples:
        for label, sample in samples.items():
            body_parts.setdefault(label, set()).add(BODY_PART_TO_LABEL.get(sample.get("body_part")))
    g = globals_of(result.raw_local.astype(np.float64))
    ori = result.calibrated_ori[np.clip(result.represented_frames, 0, result.n_frames - 1)].astype(np.float64)
    labels = result.labels
    report = {"labelToTrackerKey": rec.label_to_key,
              "slotBodyParts": {k: sorted(str(x) for x in v) for k, v in body_parts.items()},
              "slotBodyPartsConsistent": all(v == {k} for k, v in body_parts.items())}
    for a, b in (("lw", "rw"), ("ls", "rs")):
        for label, other in ((a, b), (b, a)):
            axis = np.asarray(BONE_AXIS[label])
            model = g[:, LABEL_JOINT[label]] @ axis
            same = angle_between(model, ori[:, labels.index(label)] @ axis)
            cross = angle_between(model, ori[:, labels.index(other)] @ np.asarray(BONE_AXIS[other]))
            report[f"{label}_model_vs_measured_deg"] = {"same": round(float(np.median(same)), 1),
                                                        "crossSide": round(float(np.median(cross)), 1)}
    return report


def reproduction_check(rows, result):
    frames, raw, _ = recorded_outputs(rows)
    if not len(frames):
        return {"available": False}
    lookup = {int(f): i for i, f in enumerate(result.represented_frames)}
    pairs = []
    for i, f in enumerate(frames):        # offline infers on the other frame parity half of the time
        for d in (0, -1, 1):
            if int(f) + d in lookup:
                pairs.append((i, lookup[int(f) + d]))
                break
    if not pairs:
        return {"available": False}
    live = np.asarray([raw[i] for i, _ in pairs])
    off = np.asarray([result.raw_local[j] for _, j in pairs])
    t = np.asarray([result.represented_frames[j] / 60.0 for _, j in pairs])
    diff = np.degrees(Rotation.from_matrix((np.swapaxes(live, -1, -2) @ off).reshape(-1, 3, 3)).magnitude()).reshape(len(pairs), 24)
    joints = [1, 2, 4, 5, 16, 17, 18, 19]
    out = {"available": True, "matchedOutputs": len(pairs)}
    for name, mask in (("first10s", t < 10), ("after10s", t >= 10)):
        if mask.any():
            out[name] = {"hipKneeShoulderElbow_p50_deg": round(float(np.median(diff[mask][:, joints])), 2),
                         "p95_deg": round(float(np.percentile(diff[mask][:, joints], 95)), 2)}
    return out


def summarise(result, baseline, waist):
    base = limb_angles(baseline.final_local.astype(np.float64), waist)
    fixed = limb_angles(result.final_local.astype(np.float64), waist)
    raw = limb_angles(result.raw_local.astype(np.float64), waist)
    stats = {}
    pct = lambda x: [round(float(v), 1) for v in np.percentile(x, (10, 50, 90))]
    for side in ("L", "R"):
        stats[side] = {}
        for key in ("thigh", "knee", "shoulder", "elbow"):
            stats[side][key] = {
                "rawNetwork_p10_50_90": pct(raw[side][key]),
                "baselineFinal_p10_50_90": pct(base[side][key]),
                "fixedFinal_p10_50_90": pct(fixed[side][key]),
                "baselineJump_p95_max_deg_per_output": [round(float(np.percentile(step_jumps(base[side][key]), 95)), 2),
                                                        round(float(step_jumps(base[side][key]).max()), 2)],
                "fixedJump_p95_max_deg_per_output": [round(float(np.percentile(step_jumps(fixed[side][key]), 95)), 2),
                                                     round(float(step_jumps(fixed[side][key]).max()), 2)],
            }
    return stats, raw, base, fixed


def lowering_checks(result, raw, fixed):
    """Frames where post-processing lowered a thigh by >15 deg without extension evidence."""
    out = {}
    for side, name in (("L", "left"), ("R", "right")):
        bad = 0
        uncertain_applied = 0
        for i, d in enumerate(result.diagnostics):
            leg = d["deadbugLegs"].get(name, {}) if d["deadbugLegs"].get("enabled") else {}
            lowered = raw[side]["thigh"][i] - fixed[side]["thigh"][i] > 15.0
            # "returning" glides up from an extended estimate: legitimately below a network that already shows flexed
            if lowered and leg.get("phase") in ("flexed", "unknown", None):
                bad += 1
            if leg.get("applied", 0) > 0 and leg.get("confidence", 0) < 0.35:
                uncertain_applied += 1
        out[name] = {"loweredWithoutExtensionEvidence": bad, "appliedWhileConfidenceBelow0.35": uncertain_applied}
    return out


def decision_table(result, waist_rot):
    """Every start_extend/start_return/segment decision, with which forearm was overhead then."""
    rows = []
    labels = result.labels
    ori = result.calibrated_ori
    head = -(-waist_rot[:, :, 1])
    for side, events in result.leg_events.items():
        for e in events:
            if e["event"] not in ("start_extend", "start_return", "segment_solved", "segment_no_thigh_motion",
                                  "segment_infeasible", "segment_rejected", "segment_contradicts_prior",
                                  "segment_ambiguous"):
                continue
            f = min(int(e["frame"]), len(ori) - 1)
            arm = {}
            for label in ("lw", "rw"):
                v = ori[max(0, f - 30):f + 31, labels.index(label)] @ np.asarray(BONE_AXIS[label])
                h = head[max(0, f - 30):f + 31]
                arm[label] = round(float(np.max((v * h).sum(-1))), 2)
            rows.append({"side": side, **e, "forearmTowardHeadMax(±0.5s)": arm})
    return sorted(rows, key=lambda r: r["frame"])


def write_csv(path, result, baseline, raw, base, fixed):
    labels = result.labels
    ori = result.calibrated_ori
    rec_q = Rotation.from_matrix(ori.reshape(-1, 3, 3)).as_quat().reshape(ori.shape[0], ori.shape[1], 4)
    speed = np.zeros(ori.shape[:2])
    rel = np.swapaxes(ori[:-1], -1, -2) @ ori[1:]
    speed[1:] = np.degrees(Rotation.from_matrix(rel.reshape(-1, 3, 3)).magnitude()).reshape(rel.shape[:2]) * 60.0
    fields = ["outputIndex", "inputFrame", "representedFrame", "lagFrames", "t_s"]
    for label in labels:
        fields += [f"{label}_calibrated_qxyzw", f"{label}_deg_per_s"]
    for side in ("L", "R"):
        for key in ("thigh", "knee", "shoulder", "elbow"):
            fields += [f"{side}_{key}_raw", f"{side}_{key}_baseline", f"{side}_{key}_fixed"]
        fields += [f"{side}_phase", f"{side}_confidence", f"{side}_applied", f"{side}_estimate_deg", f"{side}_reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, frame in enumerate(result.represented_frames):
            f = int(np.clip(frame, 0, result.n_frames - 1))
            row = {"outputIndex": i, "inputFrame": int(result.input_frames[i]), "representedFrame": int(frame),
                   "lagFrames": result.lag, "t_s": round(frame / 60.0, 3)}
            for j, label in enumerate(labels):
                row[f"{label}_calibrated_qxyzw"] = " ".join(f"{v:.4f}" for v in rec_q[f, j])
                row[f"{label}_deg_per_s"] = round(float(speed[f, j]), 1)
            for side, name in (("L", "left"), ("R", "right")):
                for key in ("thigh", "knee", "shoulder", "elbow"):
                    row[f"{side}_{key}_raw"] = round(float(raw[side][key][i]), 1)
                    row[f"{side}_{key}_baseline"] = round(float(base[side][key][i]), 1)
                    row[f"{side}_{key}_fixed"] = round(float(fixed[side][key][i]), 1)
                leg = result.diagnostics[i]["deadbugLegs"].get(name, {})
                row[f"{side}_phase"] = leg.get("phase", "off")
                row[f"{side}_confidence"] = leg.get("confidence", 0.0)
                row[f"{side}_applied"] = leg.get("applied", 0.0)
                row[f"{side}_estimate_deg"] = leg.get("phi")
                row[f"{side}_reason"] = leg.get("reason", "")
            writer.writerow(row)


def diagnose(name, out_dir=None, model=None):
    path = resolve_recording_path(name)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = model or rows[0].get("model")
    loaded = resolve_waist_model(model)
    print(f"{path.name}: replaying {model} (baseline: dead-bug legs off) ...", file=sys.stderr, flush=True)
    baseline = replay(path, model, leg_mode="off", loaded=loaded)
    loaded = resolve_waist_model(model)
    print(f"{path.name}: replaying {model} (current chain) ...", file=sys.stderr, flush=True)
    result = replay(path, model, leg_mode="auto", loaded=loaded)
    waist_rot = result.calibrated_ori[:, result.labels.index("waist")].astype(np.float64)
    waist = waist_rot[np.clip(result.represented_frames, 0, result.n_frames - 1)]
    stats, raw, base, fixed = summarise(result, baseline, waist)
    summary = {
        "recording": path.name, "model": model, "lagFrames": result.lag, "inferEvery": result.infer_every,
        "frames": result.n_frames, "outputs": len(result.represented_frames),
        "mapping": mapping_check(path, rows, result),
        "liveVsOfflineRawOutput": reproduction_check(rows, baseline),
        "angles": stats,
        "lowering": lowering_checks(result, raw, fixed),
        "decisions": decision_table(result, waist_rot),
    }
    out_dir = Path(out_dir or path.parent)
    csv_path = out_dir / f"{path.stem}_deadbug_frames.csv"
    json_path = out_dir / f"{path.stem}_deadbug_summary.json"
    write_csv(csv_path, result, baseline, raw, base, fixed)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary, csv_path, json_path, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--model", default=None, help="override the recording's model")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    for name in args.recordings:
        summary, csv_path, json_path, _ = diagnose(name, args.out_dir, args.model)
        print(json.dumps({k: v for k, v in summary.items() if k != "decisions"}, ensure_ascii=False, indent=1))
        for d in summary["decisions"]:
            if d["event"] in ("start_extend", "start_return", "segment_solved"):
                keep = {k: d[k] for k in ("side", "t", "event") if k in d}
                keep.update({k: d[k] for k in ("translation", "speed", "shankDip", "displacementFeetward",
                                               "displacementUp", "phiStart", "phiEnd", "confidence",
                                               "forearmTowardHeadMax(±0.5s)") if k in d})
                print("  ", keep)
        print(f"wrote {csv_path}\n      {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
