"""Lying-cat contact rule analysis: recorded live vs v3 (420afe8) vs current/new rules.

1. ``capture``: one FlowHMR replay per recording; the inputs of every
   ``LyingCatContact.apply`` call (pre-rule local pose, measured sensors, dt)
   are cached.  The rule never feeds back into the model, so any rule variant
   can then be re-applied offline on exactly the same inputs.
2. ``metrics``: knee/ankle heights (render convention: lowest mesh vertex on
   y=0), knee frame-to-frame displacement, knee flexion, shank angular step,
   per-joint rotation differences (raw->live, variant->live), and the arm chain
   local rotations (forearm-in-upper-arm, wrist roll in forearm, hand).
3. ``video``: side-by-side recorded live / variants.

Usage (from the workspace root, PYTHONPATH=runtime/python):
    python -m analysis.lyingcat_contact_report report  lyingcat_20261120_230042.jsonl ...
    python -m analysis.lyingcat_contact_report video   --variant knee_ik ...
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np

from production.arm_prior import PARENTS, _globals
from production.paths import paths

MODEL = "waist_imudino_stream_lyingcat_flowhmr_20261118"
CACHE = paths.recordings / "action_eval_20261119" / "_lyingcat_cache"
OUT_ROOT = paths.recordings / "action_eval_20261119"

VARIANTS = {
    "v3": ("analysis.lyingcat_variants.v3_pelvis", "LyingCatContact", {}),
    "v3_420afe8": ("analysis.lyingcat_variants.v3_420afe8", "LyingCatContact", {}),
    "current": ("production.lyingcat_contact", "LyingCatContact", {}),
    "v7_hand_anchor": ("analysis.lyingcat_variants.v7_hand_anchor", "LyingCatContact", {}),
    "v7b": ("analysis.lyingcat_variants.v7b_hand_anchor_height", "LyingCatContact", {}),
    "v8": ("analysis.lyingcat_variants.v8_trunk_anchor", "LyingCatContact", {}),
    "v7_twist_only": ("analysis.lyingcat_variants.v7_twist_only_20261121", "LyingCatContact", {}),
    "v6_latch": ("analysis.lyingcat_variants.v6_knee_latch_20261121", "LyingCatContact", {}),
    "v6_latch_nolock": ("analysis.lyingcat_variants.v6_knee_latch_20261121", "LyingCatContact", {"no_lock": True}),
}


def _recording(name):
    path = Path(name)
    return path if path.exists() else paths.recordings / name


def _unity_to_smpl(q):
    from production.live_replay import _unity_quats_to_smpl
    return _unity_quats_to_smpl(q).astype(np.float64)


# ---------------------------------------------------------------- capture
def capture(recording, force=False):
    path = _recording(recording)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{path.stem}.npz"
    if out.exists() and not force:
        return dict(np.load(out, allow_pickle=True))
    import production.frame_chain as frame_chain
    from production.deadbug_replay import replay
    from production.lyingcat_contact import LyingCatContact as Current

    calls = []

    class Recorder(Current):
        def apply(self, local, dt, measured=None, labels=None):
            calls.append((np.asarray(local, np.float64).copy(), float(dt),
                          None if measured is None else np.asarray(measured, np.float64).copy(), list(labels or [])))
            return super().apply(local, dt, measured=measured, labels=labels)

    original = frame_chain.LyingCatContact
    frame_chain.LyingCatContact = Recorder
    try:
        result = replay(path, model_id=MODEL, start_context_frames=60, start_offset_frames=60)
    finally:
        frame_chain.LyingCatContact = original
    assert len(calls) == len(result.final_local), (len(calls), len(result.final_local))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preds = [row for row in rows if row.get("type") == "prediction"]
    data = dict(
        times=result.represented_frames.astype(float) / 60.0,
        pre=np.stack([c[0] for c in calls]), dt=np.asarray([c[1] for c in calls]),
        measured=np.stack([c[2] for c in calls]), labels=np.asarray(calls[0][3]),
        raw=result.raw_local.astype(np.float64), final_current=result.final_local.astype(np.float64),
        live_t=np.asarray([float(r["t"]) for r in preds]),
        live=np.stack([_unity_to_smpl(r["unityLocalQuaternions"]) for r in preds]),
        live_raw=np.stack([_unity_to_smpl(r["rawLocalQuaternions"]) for r in preds]),
        live_lc_weights=np.asarray([min(((r.get("diagnostics", {}).get("deadbugLegs", {}).get("lyingcat") or {})
                                         .get("weights") or {"x": 0.0}).values()) for r in preds]),
    )
    np.savez_compressed(out, **data)
    return data


def make_rule(variant):
    module, name, opts = VARIANTS[variant]
    cls = getattr(importlib.import_module(module), name)
    rule = cls()
    if opts.get("no_lock"):
        rule._lock_contact_positions = lambda *a, **k: None
    return rule


def run_rule(data, variant):
    rule = make_rule(variant)
    labels = list(data["labels"])
    out, diags = [], []
    for local, dt, measured in zip(data["pre"], data["dt"], data["measured"]):
        res, diag = rule.apply(local.copy(), float(dt), measured=measured, labels=labels)
        out.append(np.asarray(res, np.float64))
        diags.append(diag)
    return np.stack(out), diags


# ---------------------------------------------------------------- metrics
_MODEL = None


def _smpl():
    global _MODEL
    if _MODEL is None:
        from analysis.compare_video import smpl_model
        _MODEL = smpl_model()[0]
    return _MODEL


def _offsets():
    from production.bridge_contact import _rest_offsets
    return _rest_offsets()


def render_positions(local):
    """(P,24,3) joint positions in the video convention (pelvis at x=z=0, lowest vertex on y=0)."""
    import production.mesh_video as mv
    from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
    model = _smpl()
    out = []
    for pose in local:
        q = batch_smpl_matrix_to_unity_quat_xyzw(np.asarray(pose, np.float32))
        _verts, joints = mv.skin(model, q, True, return_joints=True)
        out.append(joints)
    return np.asarray(out)


def _geodesic(a, b):
    r = np.einsum("...ji,...jk->...ik", a, b)
    tr = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def _twist_deg(rot, axis):
    """Twist angle (deg) of rotation matrix ``rot`` about unit ``axis`` (swing-twist split)."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(rot).as_quat()          # x y z w
    axis = axis / np.linalg.norm(axis)
    proj = q[..., :3] @ axis
    ang = 2 * np.degrees(np.arctan2(proj, q[..., 3]))
    return (ang + 180) % 360 - 180


def _swing_deg(rot, axis):
    axis = axis / np.linalg.norm(axis)
    v = rot @ axis
    return np.degrees(np.arccos(np.clip(v @ axis, -1, 1)))


def sequence_metrics(local, offsets, mid):
    """Per-sequence statistics on the mid-action window ``mid`` (bool mask)."""
    local = np.asarray(local, np.float64)
    g = np.stack([np.stack(_globals(pose)) for pose in local])
    pos = render_positions(local)
    m = {}
    for name, joint in (("kneeL", 4), ("kneeR", 5), ("ankleL", 7), ("ankleR", 8), ("wristL", 20), ("wristR", 21)):
        y = pos[mid, joint, 1]
        m[f"{name}_h_cm"] = (100 * y.min(), 100 * np.median(y), 100 * y.max())
    for name, joint in (("kneeL", 4), ("kneeR", 5), ("wristL", 20), ("wristR", 21)):
        step = np.linalg.norm(np.diff(pos[:, joint], axis=0), axis=1)[mid[1:] & mid[:-1]]
        drift = np.linalg.norm(pos[mid, joint] - np.median(pos[mid, joint], 0), axis=1)
        m[f"{name}_step_max_cm"] = 100 * step.max()
        m[f"{name}_drift_p95_cm"] = 100 * np.percentile(drift, 95)
    for side, hip, knee, ankle, shank in (("L", 1, 4, 7, 4), ("R", 2, 5, 8, 5)):
        thigh_v = pos[:, knee] - pos[:, hip]
        shank_v = pos[:, ankle] - pos[:, knee]
        cos = np.einsum("ij,ij->i", thigh_v, shank_v) / np.linalg.norm(thigh_v, axis=1) / np.linalg.norm(shank_v, axis=1)
        flex = np.degrees(np.arccos(np.clip(cos, -1, 1)))[mid]
        m[f"kneeFlex{side}_deg"] = (flex.min(), np.median(flex), flex.max())
        sh = _geodesic(g[:-1, shank], g[1:, shank])[mid[1:] & mid[:-1]]
        m[f"shankStep{side}_max_deg"] = sh.max()
        elev = np.degrees(np.arcsin(np.clip(shank_v[:, 1] / np.linalg.norm(shank_v, axis=1), -1, 1)))[mid]
        m[f"shankElev{side}_deg"] = (elev.min(), np.median(elev), elev.max())
    # arm chain: local twist of forearm about its bone axis, wrist roll, hand
    for side, upper, fore, wrist, hand in (("L", 16, 18, 20, 22), ("R", 17, 19, 21, 23)):
        tw_f = _twist_deg(local[:, fore], offsets[wrist])[mid]
        sw_f = np.array([_swing_deg(r, offsets[wrist]) for r in local[:, fore]])[mid]
        tw_w = _twist_deg(local[:, wrist], offsets[hand])[mid]
        ang_w = _geodesic(np.eye(3)[None], local[:, wrist])[mid]
        ang_h = _geodesic(np.eye(3)[None], local[:, hand])[mid]
        tw_s = _twist_deg(local[:, upper], offsets[fore])[mid]
        m[f"shoulderTwist{side}_deg"] = (tw_s.min(), np.median(tw_s), tw_s.max())
        m[f"elbowTwist{side}_deg"] = (tw_f.min(), np.median(tw_f), tw_f.max())
        m[f"elbowTwistStep{side}_max_deg"] = np.abs(np.diff(_twist_deg(local[:, fore], offsets[wrist])))[mid[1:] & mid[:-1]].max()
        m[f"elbowBend{side}_deg"] = (sw_f.min(), np.median(sw_f), sw_f.max())
        m[f"wristTwist{side}_deg"] = (tw_w.min(), np.median(tw_w), tw_w.max())
        m[f"wristAngle{side}_deg_p95"] = np.percentile(ang_w, 95)
        m[f"handAngle{side}_deg_p95"] = np.percentile(ang_h, 95)
        fdir = np.einsum("pij,j->pi", g[:, fore], offsets[wrist])
        m[f"_fdir{side}"] = fdir / np.linalg.norm(fdir, axis=1, keepdims=True)
    m["_g"] = g
    m["_pos"] = pos
    return m


JOINT_GROUPS = {"pelvis": [0], "hips": [1, 2], "knees(shank)": [4, 5], "ankles": [7, 8], "feet": [10, 11],
                "spine1/2/3": [3, 6, 9], "upperArm": [16, 17], "forearm": [18, 19], "wrist": [20, 21],
                "hand": [22, 23]}


def rot_diff(a, b, mid, global_=False):
    """median / p95 joint rotation difference (deg) per group on ``mid``."""
    if global_:
        a = np.stack([np.stack(_globals(p)) for p in a])
        b = np.stack([np.stack(_globals(p)) for p in b])
    d = _geodesic(a, b)[mid]
    return {k: (float(np.median(d[:, v])), float(np.percentile(d[:, v], 95))) for k, v in JOINT_GROUPS.items()}


def align_live(data):
    """Index of the live prediction closest in time to each rerun output."""
    live_t = data["live_t"]
    idx = np.searchsorted(live_t, data["times"]).clip(1, len(live_t) - 1)
    left = live_t[idx - 1]
    idx = np.where(np.abs(data["times"] - left) < np.abs(live_t[idx] - data["times"]), idx - 1, idx)
    return idx


def mid_window(data, diags):
    """Four-point support latched in the reference run, excluding first 2 s and last 1 s of it."""
    w = np.asarray([min(d.get("weights", {"x": 0}).values()) for d in diags])
    t = data["times"]
    on = w > 0.85
    if not on.any():
        return np.ones(len(t), bool)
    t0, t1 = t[on].min() + 2.0, t[on].max() - 1.0
    return on & (t >= t0) & (t <= t1)


def _fmt(v):
    if isinstance(v, tuple):
        return "/".join(f"{x:.1f}" for x in v)
    return f"{v:.1f}"


def report(recordings, variants, out_json=None):
    offsets = _offsets()
    full = {}
    for rec in recordings:
        data = capture(rec)
        idx = align_live(data)
        live = data["live"][idx]
        live_raw = data["live_raw"][idx]
        runs = {name: run_rule(data, name) for name in variants}
        mid = mid_window(data, runs[variants[0]][1])
        print(f"\n=== {rec}: {len(mid)} outputs, mid window {mid.sum()} frames "
              f"({data['times'][mid].min():.1f}-{data['times'][mid].max():.1f}s)")
        seqs = {"live": live, "replay_raw": data["raw"]}
        seqs.update({name: r[0] for name, r in runs.items()})
        metrics = {name: sequence_metrics(seq, offsets, mid) for name, seq in seqs.items()}
        keys = [k for k in metrics["live"] if not k.startswith("_")]
        print(f"{'metric (min/med/max)':28s}" + "".join(f"{n:>22s}" for n in seqs))
        for k in keys:
            print(f"{k:28s}" + "".join(f"{_fmt(metrics[n][k]):>22s}" for n in seqs))
        print("-- forearm global direction vs live (deg, median/p95)")
        for n in seqs:
            for side in "LR":
                d = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", metrics[n][f"_fdir{side}"],
                                                            metrics["live"][f"_fdir{side}"]), -1, 1)))[mid]
                print(f"   {n:16s}{side}: {np.median(d):5.1f} / {np.percentile(d, 95):5.1f}")
        print("-- local rotation diff vs live (deg median/p95): live_raw->live, then each variant->live")
        pairs = [("live_raw", live_raw)] + [(n, s) for n, s in seqs.items() if n != "live"]
        print(f"{'group':14s}" + "".join(f"{n:>18s}" for n, _ in pairs))
        diffs = {n: rot_diff(s, live, mid) for n, s in pairs}
        for grp in JOINT_GROUPS:
            print(f"{grp:14s}" + "".join(f"{_fmt(diffs[n][grp]):>18s}" for n, _ in pairs))
        print("-- rule change vs replay raw (local deg median/p95)")
        print(f"{'group':14s}" + "".join(f"{n:>18s}" for n in runs))
        cd = {n: rot_diff(runs[n][0], data["raw"], mid) for n in runs}
        for grp in JOINT_GROUPS:
            print(f"{grp:14s}" + "".join(f"{_fmt(cd[n][grp]):>18s}" for n in runs))
        full[rec] = {n: {k: v for k, v in metrics[n].items() if not k.startswith("_")} for n in seqs}
        full[rec]["rotDiffVsLive"] = diffs
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(full, indent=1, default=lambda o: list(o) if isinstance(o, tuple)
                                             else float(o)), encoding="utf-8")
    return full


def video(recordings, variants, out_dir, after=8.83, workers=1):
    from analysis.compare_video import render
    titles = {"v3": "v3_pelvis 重跑", "v3_420afe8": "旧版 420afe8 重跑", "current": "新 production 重跑",
              "v6_latch": "中间版(膝锁存) 重跑"}
    for rec in recordings:
        data = capture(rec)
        idx = align_live(data)
        cols = [("recorded", "录制时实时输出 (unityLocalQuaternions)", data["live"][idx])]
        for name in variants:
            # "current" = the production replay output itself (deadbug_replay -> frame_chain -> production rule)
            seq = data["final_current"] if name == "current" else run_rule(data, name)[0]
            cols.append((name, titles.get(name, f"{name} 重跑 (FlowHMR+规则)"), seq))
        times = data["times"]
        keep = times >= after
        cols = [(k, t, l[keep]) for k, t, l in cols]
        text = {k: [[k] for _ in range(int(keep.sum()))] for k, _t, _l in cols}
        out = Path(out_dir) / f"recorded_vs_rerun_{_recording(rec).stem}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        render(out, cols, times[keep], text, f"{_recording(rec).name} | " + " vs ".join(k for k, _t, _l in cols),
               frame_key=cols[-1][0], workers=workers)
        print(out.resolve(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["capture", "report", "video"])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--variants", default="v3,current,current_nolock")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    variants = args.variants.split(",")
    if args.cmd == "capture":
        for rec in args.recordings:
            capture(rec, force=args.force)
    elif args.cmd == "report":
        report(args.recordings, variants, args.json)
    else:
        video(args.recordings, variants, args.out_dir or OUT_ROOT / "_tmp", workers=args.workers)


if __name__ == "__main__":
    main()
