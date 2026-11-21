"""Sensor correction from the prescribed exercise start pose (no separate calibration moves).

The user only holds the action's own start pose.  For each limb sensor the full orientation of the
bone in that pose is known from a template (bone direction *and* twist, e.g. palms flat on the floor
with the fingers forward), expressed relative to the trunk's head direction.  The measured hold is

    measured = W @ true @ M          W: world yaw drift (SlimeVR, no magnetometer)
                                     M: mounting error (sensor frame)

One hold gives 3 DOF for these 4 unknowns.  The yaw W absorbs as much as it can (best yaw), the
rest becomes the mounting correction M, so the corrected start pose equals the template exactly.
This is the start-pose "delta" applied at the model input: corrected_t = W^T measured_t M^T.
"""
from __future__ import annotations

import numpy as np

from production.pose_calibration import BONE_AXES, _angle_deg, _best_yaw, _rotation_between, _yaw


def _frame(x, y, z):
    """Rotation whose columns are the images of the SMPL x, y, z axes."""
    return np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)], axis=1)


# SMPL bone globals in a body frame: +z = head direction made horizontal, +y = up, +x = body left
# when facing +z.  T-pose: left forearm +x / right forearm -x along the bone, palm -y; shank -y along
# the bone, shin front +z.
_QUADRUPED = {
    # forearm vertical (elbow above wrist), hand flat with the fingers toward the head:
    # the palm side of the forearm faces the knees (-z)
    "lw": _frame((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    "rw": _frame((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    # shank flat on the floor pointing to the feet (-z), shin front down (-y)
    "ls": _frame((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "rs": _frame((1, 0, 0), (0, 0, 1), (0, -1, 0)),
}

# Lying on the back, head toward +z: chest up (+y), body-left toward -x.
_S60, _C60 = np.sin(np.radians(60.0)), np.cos(np.radians(60.0))
# Knees bent, feet flat: knee -> ankle 60 deg below level toward the feet (measured live 59-63 deg),
# shin front facing up/feetward.
_SHANK_FEET_FLAT = _frame((-1, 0, 0), (0, _S60, _C60), (0, _C60, -_S60))
_SUPINE_ARMS_AT_SIDES = {
    # forearm along the body toward the feet, palm down
    "lw": _frame((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "rw": _frame((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "ls": _SHANK_FEET_FLAT, "rs": _SHANK_FEET_FLAT,
}

def _hands_behind_head(elevation_deg: float, headward_deg: float):
    """Forearm from the elbow (out at the side, raised) to the hand under the back of the head:
    toward the body centre, `headward_deg` toward the head, `elevation_deg` (< 0: wrist below elbow);
    palm against the head, i.e. facing as far up as the bone allows."""
    e, a = np.radians(elevation_deg), np.radians(headward_deg)
    frames = {}
    for label, centre in (("lw", 1.0), ("rw", -1.0)):        # supine: body-left is -x, centre of lw is +x
        bone = np.array([centre * np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
        palm = np.array([0.0, 1.0, 0.0]) - np.sin(e) * bone
        palm /= np.linalg.norm(palm)
        x = bone if label == "lw" else -bone                 # BONE_AXES: lw +x, rw -x
        y = -palm                                            # T-pose palm faces -y
        frames[label] = _frame(x, y, np.cross(x, y))
    return frames


# Hands behind the head.  Measured on the one live hold that really was hands-behind-head
# (crunch_20261119_232340, 3-15 s, raw SlimeVR headings within 12 deg of this): forearm 37-43 deg
# below level from the raised elbow, 15-18 deg toward the head, palm normal 0.7-0.77 up.  The
# 2026-11-19 "palm down" template came from holds where the arms lay at the sides (old prompt):
# palm down, level, pointing out/feetward -- it fitted those with a ~180 deg heading, which then
# turned the hands-behind-head forearms of the crunch toward the feet (hands in front of the face).
_SUPINE_HANDS_BEHIND_HEAD = {**_hands_behind_head(-30.0, 20.0),
                             "ls": _SHANK_FEET_FLAT, "rs": _SHANK_FEET_FLAT}
_TABLETOP = {
    # forearms vertical up, palms facing each other
    "lw": _frame((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    "rw": _frame((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    # thighs vertical, shanks level toward the feet, shin front up
    "ls": _frame((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "rs": _frame((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
}

# Side bridge before the hips lift: lying on one side, head toward +z, supporting elbow under the
# shoulder with the forearm flat on the floor pointing forward (the chest side), palm down; legs
# straight in line with the body; the top arm straight along the body, hand on the outer thigh.
# Left side down: body-left -> -y (down), chest -> -x.  Right side down: body-left -> +y, chest -> +x.
_SIDE_LEFT_DOWN = {
    "lw": _frame((-1, 0, 0), (0, 1, 0), (0, 0, -1)),      # support forearm forward, palm down
    "rw": _frame((0, 0, 1), (0, 1, 0), (-1, 0, 0)),       # top arm toward the feet, palm on the thigh
    "ls": _frame((0, -1, 0), (0, 0, 1), (-1, 0, 0)),      # shank toward the feet, shin facing forward
    "rs": _frame((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
}
_SIDE_RIGHT_DOWN = {
    "rw": _frame((-1, 0, 0), (0, 1, 0), (0, 0, -1)),      # support forearm forward (+x), palm down
    "lw": _frame((0, 0, -1), (0, 1, 0), (1, 0, 0)),       # top arm toward the feet, palm on the thigh
    "ls": _frame((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "rs": _frame((0, 1, 0), (0, 0, 1), (1, 0, 0)),
}

# Start poses whose forearm twist (palm direction) is not prescribed: a raised arm's palm may face
# anywhere, and for a vertical forearm that twist is indistinguishable from the heading.  Live dead bug
# 2026-11-19: a palms-facing guess put the forearm headings +-135 deg off on both recordings.
FREE_TWIST = {("deadbug", "tabletop"): frozenset({"lw", "rw"})}

TEMPLATES = {
    ("sidebridge_left", "side_rest"): _SIDE_LEFT_DOWN,
    ("sidebridge_right", "side_rest"): _SIDE_RIGHT_DOWN,
    ("birddog", "quadruped"): _QUADRUPED,
    ("lyingcat", "quadruped"): _QUADRUPED,
    ("hipthrust", "supine"): _SUPINE_ARMS_AT_SIDES,
    ("crunch", "supine"): _SUPINE_HANDS_BEHIND_HEAD,
    ("deadbug", "tabletop"): _TABLETOP,
}
# Mounting residual after the heading: its swing moves where the bone points (live mounting errors
# 2-21 deg); its twist is mostly the palm's orientation (live hip thrust: right palm ~45 deg off "down").
# Crunch: arms at the sides instead of hands behind the head shows up as a 150-177 deg twist (palm down vs up).
REJECT_HINTS = {("crunch", "supine"): "请双手抱在头后、手肘向两侧打开再试（双臂放体侧不能用于卷腹校准）"}
MAX_SWING_DEG = 40.0
MAX_TWIST_DEG = 90.0
MAX_ELEVATION_DEG = 25.0    # bone elevation is heading-free: the user is not in the start pose if off


def elevation_error(label: str, bone: np.ndarray, measured: np.ndarray) -> float:
    """|elevation(measured bone axis) - elevation(template bone axis)| in degrees (yaw-invariant)."""
    axis = BONE_AXES[label]
    el = lambda v: np.degrees(np.arcsin(np.clip(v[1], -1.0, 1.0)))
    return float(abs(el(measured @ axis) - el(bone @ axis)))


def pose_matches(template: dict, holds: dict) -> bool:
    return all(elevation_error(label, bone, holds[label]) <= MAX_ELEVATION_DEG
               for label, bone in template.items() if label in holds)


HOLD_FRAMES = 30            # 0.5 s still window
STILL_DEG = 6.0


def find_hold(samples: np.ndarray, labels, template: dict):
    """Mean raw orientations of the latest still 0.5 s window of `samples` (T, n, 3, 3) in which the
    limbs are in the start pose (bone elevations match the template); None if there is none."""
    from scipy.spatial.transform import Rotation
    samples = np.asarray(samples)
    for end in range(len(samples), HOLD_FRAMES - 1, -5):
        window = samples[end - HOLD_FRAMES:end]
        rotations = [Rotation.from_matrix(window[:, i]) for i in range(len(labels))]
        if any(np.degrees((r.mean().inv() * r).magnitude().max()) > STILL_DEG for r in rotations):
            continue
        holds = {label: rotations[i].mean().as_matrix() for i, label in enumerate(labels)}
        if pose_matches(template, holds):
            return holds
    return None


def fit_calibrator(calibrator, samples, action: str, start_pose: str):
    """Live and replay entry point: fit the limb corrections from the raw `samples` of the start hold
    and commit them to `calibrator` (headings/corrections) only if every limb passes.
    Returns (passed, message)."""
    template = TEMPLATES[(action, start_pose)]
    labels = calibrator.labels
    if "waist" not in labels:
        return False, f"start pose fit ({action}/{start_pose}): no waist sensor"
    holds = find_hold(samples, labels, template)
    if holds is None:
        return False, f"start pose fit ({action}/{start_pose}): 未找到符合起始姿势的稳定0.5秒，请摆好起始姿势保持不动"
    fitted, rejected = fit(template, holds, holds["waist"], FREE_TWIST.get((action, start_pose), frozenset()))
    report = ", ".join(f"{label}: 朝向{v[2]:+.0f}°/倾斜{v[3]:.0f}°/扭转{v[4]:.0f}°"
                       + (" REJECTED" if label in rejected else "") for label, v in fitted.items())
    if rejected:
        hint = REJECT_HINTS.get((action, start_pose), "请重试")
        return False, f"start pose fit ({action}/{start_pose}): {report}；起始姿势与模板不符，{hint}"
    for label, (heading, mount, *_rest) in fitted.items():
        i = labels.index(label)
        calibrator.headings[i], calibrator.corrections[i] = heading, mount
    return True, f"start pose fit ({action}/{start_pose}): {report}"


def head_yaw(waist: np.ndarray) -> np.ndarray:
    """Yaw rotation taking the template +z (head direction) to the measured trunk's head direction."""
    spine = waist @ np.array([0.0, 1.0, 0.0])
    return _yaw(float(np.arctan2(spine[0], spine[2])))


def swing_twist_deg(label: str, mount: np.ndarray):
    """Split a sensor-frame correction into the swing of the bone axis (changes where the bone points)
    and the twist about it (palm orientation only)."""
    axis = BONE_AXES[label]
    swing = _rotation_between(axis, mount @ axis)
    return _angle_deg(swing), _angle_deg(swing.T @ mount)


def fit(template: dict, holds: dict, waist: np.ndarray, free_twist=frozenset()):
    """holds: label -> mean measured (uncorrected) orientation of the start hold.
    Returns {label: (heading, correction, yaw_deg, swing_deg, twist_deg)} and the rejected labels.

    Level bone: the heading aligns the bone direction (its twist cannot bias the yaw).  Steep bone
    (vertical forearm): direction carries no heading, so the yaw takes the whole rotation about the
    vertical -- there yaw drift and twist are the same rotation.
    `free_twist` labels (a raised forearm whose palm may face anywhere): only the bone direction is
    corrected (minimal swing); the SlimeVR heading is kept and the first reach aligns it (reach_heading)."""
    facing = head_yaw(waist)
    out, rejected = {}, []
    for label, bone in template.items():
        if label not in holds:
            continue
        target = facing @ bone
        measured = holds[label]
        axis = BONE_AXES[label]
        want, have = target @ axis, measured @ axis
        if label in free_twist:
            heading = np.eye(3)
            mount = _rotation_between(axis, measured.T @ want)
            swing_deg, twist_deg = _angle_deg(mount), 0.0
            out[label] = (heading, mount, 0.0, swing_deg, twist_deg)
            if swing_deg > MAX_SWING_DEG or elevation_error(label, bone, measured) > MAX_ELEVATION_DEG:
                rejected.append(label)
            continue
        if abs(want[1]) < np.sin(np.radians(45.0)) and abs(have[1]) < np.sin(np.radians(45.0)):
            heading = _yaw(float(np.arctan2(want[0], want[2]) - np.arctan2(have[0], have[2])))
        else:
            heading = _yaw(_best_yaw([target @ measured.T]))
        mount = measured.T @ heading.T @ target
        swing_deg, twist_deg = swing_twist_deg(label, mount)
        yaw_deg = float(np.degrees(np.arctan2(heading[0, 2], heading[0, 0])))
        out[label] = (heading, mount, yaw_deg, swing_deg, twist_deg)
        if (swing_deg > MAX_SWING_DEG or twist_deg > MAX_TWIST_DEG
                or elevation_error(label, bone, measured) > MAX_ELEVATION_DEG):
            rejected.append(label)
    return out, rejected
