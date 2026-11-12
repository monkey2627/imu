"""Synthetic check of the lying-game calibration (docs/09, PoseCalibrator.request_pose_hold).

True SMPL bone orientations of each hold get known errors: a world heading on every forearm/shank, a
roll mount error on the waist, ~5 deg random mount tilts.  Each flow (dead bug, hip thrust, left side
bridge) is replayed through PoseCalibrator; then the bone pointing error is measured in held-out
standing poses.  A last case puts the arms 30 deg out from the body in the first hold (a user error)
to see whether the later checks flag it.

python -m analysis.calibration.pose_hold_synthetic   (from realUse/)
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from production.pose_calibration import BONE_AXES, PoseCalibrator

LABELS = ["lw", "rw", "ls", "rs", "waist"]
R = lambda ax, deg: Rotation.from_euler(ax, deg, degrees=True).as_matrix()
I3 = np.eye(3)
SUPINE = R("x", -90)                       # spine -> -z (head), chest -> +y (up)
SIDE_LEFT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # left side down, chest +x


def pose(lw, rw, ls, rs, waist, facing):
    f = R("y", facing)
    return np.stack([f @ m for m in (lw, rw, ls, rs, waist)])


def holds(facing, arms_out=0.0):
    out_l, out_r = R("y", arms_out), R("y", -arms_out)          # arms swung away from the body
    arms_side = (out_l @ R("y", -90), out_r @ R("y", 90))       # along the body toward the feet, palms down
    knees_up = R("x", 20)                                       # knee->ankle steeply down toward the hips
    legs_straight = (SUPINE, SUPINE)                            # knee->ankle toward the feet
    return {
        "supine_arms_side": pose(*arms_side, knees_up, knees_up, SUPINE, facing),
        "supine_legs_straight": pose(*arms_side, *legs_straight, SUPINE, facing),
        "deadbug_tabletop": pose(R("z", 90), R("z", -90), SUPINE, SUPINE, SUPINE, facing),
        "deadbug_reach_left": pose(R("y", 90) @ R("z", 15), R("z", -90), SUPINE, SUPINE, SUPINE, facing),
        "deadbug_reach_right": pose(R("z", 90), R("y", -90) @ R("z", -15), SUPINE, SUPINE, SUPINE, facing),
        "hipthrust_start": pose(*arms_side, knees_up, knees_up, SUPINE, facing),
        "sidebridge_left_start": pose(I3, R("y", 90), R("z", -90), R("z", -90), SIDE_LEFT, facing),
    }


def held_out(facing):
    up = I3
    return {"stand arms fwd": pose(R("y", -90), R("y", 90), up, up, up, facing),
            "stand arms side": pose(I3, I3, up, up, up, facing),
            "sit shanks fwd": pose(R("z", -90), R("z", 90), R("x", -90), R("x", -90), up, facing)}


FLOWS = {"dead bug": ["supine_arms_side", "deadbug_tabletop", "deadbug_reach_left", "deadbug_reach_right"],
         "hip thrust": ["supine_legs_straight", "hipthrust_start"],
         "side bridge": ["supine_legs_straight", "sidebridge_left_start"]}


def run(flow, heading, mounts, facing=37.0, arms_out=0.0, seed=0):
    rng = np.random.default_rng(seed)
    corrupt = lambda p: np.stack([R("y", heading[i]) @ p[i] @ mounts[i] for i in range(5)])
    cal = PoseCalibrator(LABELS)
    poses = holds(facing, arms_out)
    messages = []
    for step in FLOWS[flow]:
        truth = poses[step]
        for _ in range(240):                                    # 4 s hold
            noise = [Rotation.from_rotvec(rng.normal(0, np.radians(0.5), 3)).as_matrix() for _ in range(5)]
            cal.correct(torch.from_numpy(np.stack([n @ m for n, m in zip(noise, corrupt(truth))])))
            cal.advance(1)
        cal.request_pose_hold(step, 4.0)
        messages.append(cal.advance(1))
    errors = {}
    tests = held_out(facing + 15)
    for name, truth in tests.items():
        meas = corrupt(truth)
        row = []
        for i, label in enumerate(("lw", "rw", "ls", "rs")):
            corrected = cal.headings[i] @ meas[i] @ cal.corrections[i]
            row.append(np.degrees(np.arccos(np.clip((corrected @ BONE_AXES[label]) @ (truth[i] @ BONE_AXES[label]), -1, 1))))
        errors[name] = row
    w = 4
    waist_err = np.degrees(Rotation.from_matrix((cal.headings[w] @ corrupt(tests["stand arms fwd"])[w] @ cal.corrections[w]).T
                                                @ tests["stand arms fwd"][w]).magnitude())
    return messages, errors, waist_err


def main():
    rng = np.random.default_rng(3)
    mounts = [Rotation.from_rotvec(rng.normal(0, np.radians(3), 3)).as_matrix() for _ in range(4)] + [R("y", 10)]                                       # waist roll error about the spine
    heading = [25, -115, 30, -20, 0]                             # forearm / shank world heading errors
    for flow in FLOWS:
        for arms_out in (0.0, 30.0):
            if arms_out and flow != "dead bug":
                continue
            msgs, errors, waist_err = run(flow, heading, mounts, arms_out=arms_out)
            print(f"\n### {flow}" + (f"  (USER ERROR: arms {arms_out:.0f} deg out in the first hold)" if arms_out else ""))
            for m in msgs:
                print("  " + m)
            print("  held-out pointing error after calibration (deg)  lw / rw / ls / rs:")
            for name, row in errors.items():
                print(f"    {name:16s} " + " / ".join(f"{v:5.1f}" for v in row))
            print(f"    waist rotation error (standing): {waist_err:.1f} deg (injected: 10 deg roll, heading 0)")


if __name__ == "__main__":
    main()
