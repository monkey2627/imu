"""Synthetic check of the bird-dog calibration: old swing-only solver vs. the reach-heading solver.

True SMPL bone orientations of the three holds (base, left reach, right reach) get a known
world-heading error on each forearm (the error type measured live) plus small mount tilts.
After calibration, the forearm pointing error is measured in the calibration holds and in
held-out poses (standing arms forward / hanging / sideways, and a sideways body facing).

python -m analysis.calibration.bird_dog_calib_synthetic   (from realUse/)
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from production import pose_calibration as new
from analysis.calibration import pose_calibration_before_reach_heading as old

LABELS = ["lw", "rw", "ls", "rs", "waist"]
R = lambda ax, deg: Rotation.from_euler(ax, deg, degrees=True).as_matrix()
AXES = new.BONE_AXES


def pose(root, lw, rw, ls, rs, facing):
    f = R("y", facing)
    return np.stack([f @ lw, f @ rw, f @ ls, f @ rs, f @ root])


def holds(facing):
    quad, back = R("x", 90), R("x", 90)              # trunk level (head +z), shank pointing back
    base = pose(quad, R("z", -90), R("z", 90), back, back, facing)
    left = pose(quad, R("y", -90), R("z", 90), back, back, facing)
    right = pose(quad, R("z", -90), R("y", 90), back, back, facing)
    return base, left, right


def held_out(facing):
    up = np.eye(3)
    return {"stand fwd": pose(up, R("y", -90), R("y", 90), up, up, facing),
            "stand hang": pose(up, R("z", -90), R("z", 90), up, up, facing),
            "stand side": pose(up, np.eye(3), np.eye(3), up, up, facing),
            "quad fwd palm-in": pose(R("x", 90), R("y", -90) @ R("x", 90), R("y", 90) @ R("x", -90),
                                     R("x", 90), R("x", 90), facing)}


def run(module, err_heading, err_mount, facing=20.0, seed=0, calibrate=True):
    rng = np.random.default_rng(seed)
    corrupt = lambda p: np.stack([R("y", err_heading[i]) @ p[i] @ err_mount[i] for i in range(5)])
    cal = module.PoseCalibrator(LABELS)
    cal.start_bird_dog_sequence()
    msg = None
    for poses, request in zip(holds(facing), (cal.request_bird_dog_base,
                                              lambda: cal.request_bird_dog_reach(True),
                                              lambda: cal.request_bird_dog_reach(False))):
        for _ in range(60):
            noise = [Rotation.from_rotvec(rng.normal(0, np.radians(0.5), 3)).as_matrix() for _ in range(5)]
            cal.correct(torch.from_numpy(np.stack([n @ m for n, m in zip(noise, corrupt(poses))])))
            cal.advance(1)
        if calibrate:
            request()
        msg = cal.advance(1) or msg
    names = ["quad base", "quad L-reach", "quad R-reach"]
    tests = dict(zip(names, holds(facing)))
    tests.update(held_out(facing))
    tests.update({k + " (face +90)": v for k, v in held_out(facing + 90).items()})
    result = {}
    for name, truth in tests.items():
        meas = corrupt(truth)
        errs = []
        for i, label in enumerate(("lw", "rw")):
            corrected = cal.headings[i] @ meas[i] @ cal.corrections[i]
            errs.append(np.degrees(np.arccos(np.clip((corrected @ AXES[label]) @ (truth[i] @ AXES[label]), -1, 1))))
        result[name] = errs
    return msg, result


def main():
    rng = np.random.default_rng(1)
    mounts = [Rotation.from_rotvec(rng.normal(0, np.radians(5), 3)).as_matrix() for _ in range(5)]
    cases = {"heading L+22/R-33 (this wear)": [22, -33, 15, -15, 0],
             "heading L+40/R-70 (worst seen)": [40, -70, 15, -15, 0],
             "no heading error": [0, 0, 0, 0, 0]}
    for case, heading in cases.items():
        print(f"\n### {case}, plus ~5 deg random mount tilt; forearm pointing error L/R (deg)")
        out = {}
        for name, module in (("old", old), ("new", new)):
            msg, out[name] = run(module, heading, mounts)
            print(f"  {name}: {msg[:230] if msg else None}")
        raw = run(new, heading, mounts, calibrate=False)[1]
        for test in out["new"]:
            o, n = out["old"][test], out["new"][test]
            r = raw[test]
            print(f"  {test:28s} raw {r[0]:5.1f} / {r[1]:5.1f}   old {o[0]:5.1f} / {o[1]:5.1f}   new {n[0]:5.1f} / {n[1]:5.1f}")


if __name__ == "__main__":
    main()
