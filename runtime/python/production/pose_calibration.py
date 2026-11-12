"""Per-set mounting calibration from the coach's start pose, and sensed-bone anchoring.

1. Calibration: when a set starts, the user holds the action's start pose and
   Unity sends the coach's SMPL global orientation of every sensed bone (the
   inverse of the avatar retarget, so it shares the display's conventions).
   With R_meas the raw bone orientation of a still sensor and R_coach its
   target, a world yaw Y (from the waist) absorbs the facing difference.
   - "heading" (default): each sensor gets a world-yaw correction H so that
     H R_meas matches Y R_coach about the vertical axis. Tracker heading
     errors (drift, a yaw reset taken in the wrong pose) are world-frame, so
     this stays valid when the pose changes, e.g. kneeling -> standing.
   - "mount": a right-multiplied mounting correction C = R_meas^T Y R_coach.
     It also fixes tilt, but only exactly at the calibration pose when the
     actual error is a heading error, so it is not the default.
   Corrections persist until the next set. Moving sensors and corrections
   above a limit are skipped (the user's pose, not the sensor, differs).
2. Anchoring: the sensed bones' global orientations in the model output are
   replaced by the corrected measurement of the frame the output represents;
   every other joint keeps the model's local rotation.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from production.rotmat import UNITY_SMPL_BASIS

LABEL_JOINT = {"lw": 18, "rw": 19, "ls": 4, "rs": 5, "waist": 3, "pelvis": 0}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SETTLE_FRAMES = 15          # wait after the request so the window spans the request moment
WINDOW_FRAMES = 30          # 0.5 s at 60 Hz
MAX_STILL_DEG = 6.0         # a sensor moving more than this inside the window is not calibrated
MAX_CORRECTION_DEG = 45.0   # mount mode: larger corrections mean a pose mismatch, not a mounting error
MAX_HEADING_DEG = 60.0      # arms-forward: forearm heading errors of ~50 deg were measured live
MAX_BIRD_DOG_HEADING_DEG = 120.0  # bird-dog reaches can expose a sensor pointing sideways
MAX_BIRD_DOG_MOUNT_STEP_DEG = 45.0
MAX_BIRD_DOG_FIT_ERROR_DEG = 20.0
MAX_BIRD_DOG_ARM_HEADING_DEG = 180.0  # any: 20-70 deg is typical, ~115 deg was measured on a right forearm
MAX_BIRD_DOG_FOREARM_TILT_DEG = 30.0  # base-pose forearm vs vertical: 10-14 deg measured live (2026-11-12)
REACH_SEARCH_FRAMES = 480        # the reaching arm's hold is searched over the reach step (8 s)
# Against the coach's start pose only small heading fixes are trusted: after a standing yaw reset the
# shanks are close, and larger differences are the user's pose (e.g. feet direction), not the sensor.
MAX_POSE_HEADING_DEG = 20.0
MAX_STAND_TILT_DEG = 40.0         # waist/shank correction at the standing pose
MAX_TWO_POSE_FIT_DEG = 15.0       # rms disagreement left after the forearm two-pose fit
# Known SMPL bone orientations (facing +z) of the two calibration poses.
_RZ = lambda deg: Rotation.from_euler("z", deg, degrees=True).as_matrix()
_RY = lambda deg: Rotation.from_euler("y", deg, degrees=True).as_matrix()
ARMS_DOWN = {"lw": _RZ(-90.0), "rw": _RZ(90.0)}       # hanging, palms to the thighs
ARMS_FORWARD = {"lw": _RY(-90.0), "rw": _RY(90.0)}    # straight forward, level, palms down
MAX_PARALLEL_HEADING_DEG = 45.0   # shank heading error measured live: ~35 deg inward on each side
MAX_LEVEL_ELEVATION_DEG = 45.0   # a bone steeper than this has no reliable heading (only twist)
# Unit vector along each sensed bone in its SMPL frame (elbow->wrist, knee->ankle).
BONE_AXES = {"lw": np.array([1.0, 0.0, 0.0]), "rw": np.array([-1.0, 0.0, 0.0]),
             "ls": np.array([0.0, -1.0, 0.0]), "rs": np.array([0.0, -1.0, 0.0])}
SERVER_TO_SMPL = np.diag([-1.0, 1.0, -1.0])
# Known holds of the lying games (docs/09).  Targets are body-relative: "+h"/"-h" toward the head/feet
# (the waist's spine axis, made horizontal), "+c" the chest normal made horizontal, "up" vertical;
# "roll" levels the waist's left-right axis (lying on the back).  "solve" sets a world heading (bones)
# or the roll (waist); "check" only reports the remaining error.
KNOWN_HOLDS = {
    "supine_arms_side": {"waist": ("roll", "solve"), "lw": ("-h", "solve"), "rw": ("-h", "solve")},
    "supine_legs_straight": {"waist": ("roll", "solve"), "lw": ("-h", "solve"), "rw": ("-h", "solve"),
                             "ls": ("-h", "solve"), "rs": ("-h", "solve")},
    "deadbug_tabletop": {"ls": ("-h", "solve"), "rs": ("-h", "solve"), "lw": ("up", "check"), "rw": ("up", "check")},
    "deadbug_reach_left": {"lw": ("+h", "check")},
    "deadbug_reach_right": {"rw": ("+h", "check")},
    "hipthrust_start": {"lw": ("-h", "check"), "rw": ("-h", "check")},
    "sidebridge_left_start": {"lw": ("+c", "check"), "rw": ("-h", "check")},
}
HOLD_HISTORY_FRAMES = 720         # 12 s of raw input kept for the step-long hold search
POSE_HOLD_CHECK_DEG = 25.0        # a check further off than this asks the user to recalibrate
POSE_HOLD_UP_CHECK_DEG = 20.0
MAX_POSE_HOLD_ROLL_DEG = 25.0


def unity_quat_to_smpl(quat_xyzw) -> np.ndarray:
    unity = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    return UNITY_SMPL_BASIS @ unity @ UNITY_SMPL_BASIS


def _angle_deg(matrix: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(matrix).magnitude()))


def _azimuth(vector: np.ndarray) -> float:
    return float(np.arctan2(vector[0], vector[2]))     # about the y (up) axis, matching _yaw


def _elevation(vector: np.ndarray) -> float:
    return float(np.degrees(np.arcsin(np.clip(vector[1] / np.linalg.norm(vector), -1.0, 1.0))))


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _best_yaw(offsets) -> float:
    """Angle of the y-axis rotation Y maximising sum tr(Y^T M) over the given matrices."""
    return float(np.arctan2(sum(m[0, 2] - m[2, 0] for m in offsets), sum(m[0, 0] + m[2, 2] for m in offsets)))


def _yaw(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3)
        reference = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, reference)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    axis = cross / cross_norm
    return Rotation.from_rotvec(axis * np.arctan2(cross_norm, dot)).as_matrix()


class PoseCalibrator:
    def __init__(self, labels: List[str]) -> None:
        self.labels = list(labels)
        self.calibration_enabled = True
        self.anchor_enabled = False   # anchoring shanks to the measurement crossed the ankles live
        self.mode = "heading"
        self.headings = np.tile(np.eye(3), (len(self.labels), 1, 1))      # left (world) yaw corrections
        self.corrections = np.tile(np.eye(3), (len(self.labels), 1, 1))   # right (mounting) corrections
        self._raw = deque(maxlen=2 * WINDOW_FRAMES)
        self._history = deque(maxlen=HOLD_HISTORY_FRAMES)
        self._corrected = deque(maxlen=128)
        self._pending = None
        self._stand = None          # pose-A window means, kept for the two-pose forearm solve
        self._bird_dog_base = None
        self._bird_dog_reaches = {}
        self._bird_dog_arm_holds = {}
        self._bird_dog_samples = None
        self._bird_dog_gravity = None

    def reset(self) -> None:
        """New input source or wearing: previous corrections no longer apply."""
        self.headings = np.tile(np.eye(3), (len(self.labels), 1, 1))
        self.corrections = np.tile(np.eye(3), (len(self.labels), 1, 1))
        self._raw.clear()
        self._history.clear()
        self._corrected.clear()
        self._pending = None
        self._stand = None
        self._bird_dog_base = None
        self._bird_dog_reaches = {}
        self._bird_dog_arm_holds = {}
        self._bird_dog_samples = None
        self._bird_dog_gravity = None

    def correct(self, ori: torch.Tensor) -> torch.Tensor:
        """Record the raw (n_labels, 3, 3) orientation and return the corrected one."""
        raw = ori.numpy().astype(np.float64)
        self._raw.append(raw)
        self._history.append(raw)
        if self._bird_dog_samples is not None:
            self._bird_dog_samples.append(raw.copy())
        if self.calibration_enabled:
            ori = torch.from_numpy(self.headings).to(ori) @ ori @ torch.from_numpy(self.corrections).to(ori)
        self._corrected.append(ori.numpy().copy())
        return ori

    def correct_acceleration(self, accel: torch.Tensor) -> torch.Tensor:
        """Apply the same world-yaw correction used for orientations to world acceleration."""
        if not self.calibration_enabled:
            return accel
        headings = torch.from_numpy(self.headings).to(accel)
        return torch.einsum("nij,nj->ni", headings, accel)

    def repeat_last(self, count: int) -> None:
        """Mirror window frames that re-use the last sample, keeping lag indexing aligned."""
        for _ in range(count):
            self._raw.append(self._raw[-1])
            self._history.append(self._history[-1])
            self._corrected.append(self._corrected[-1])

    def request(self, targets: Dict[str, list]) -> None:
        """Calibrate against the coach's pose; the window is centred on the request moment."""
        coach = {label: unity_quat_to_smpl(q) for label, q in targets.items() if label in self.labels}
        if not coach:
            raise ValueError("no coach targets for the current sensor labels")
        self._pending = {"kind": "coach", "coach": coach, "frames": SETTLE_FRAMES}

    def request_stand(self) -> None:
        """Pose A: standing straight, arms hanging with palms to the thighs (window = the last 0.5 s)."""
        self._pending = {"kind": "stand", "frames": 0}

    def request_arms_forward(self) -> None:
        """Pose B: both arms straight forward, level, palms down (window = the last 0.5 s)."""
        self._pending = {"kind": "arms", "frames": 0}

    def request_bird_dog_base(self) -> None:
        """Mark the neutral-pose hold for a sequence already being recorded."""
        self._bird_dog_base = None
        self._bird_dog_reaches = {}
        self._bird_dog_arm_holds = {}
        self._pending = {"kind": "bird_dog_base", "frames": 0}

    def start_bird_dog_sequence(self) -> None:
        """Start collecting all holds and transitions before the first pose command."""
        self._bird_dog_base = None
        self._bird_dog_reaches = {}
        self._bird_dog_arm_holds = {}
        self._bird_dog_samples = []
        self._bird_dog_gravity = []
        for label in ("lw", "rw", "ls", "rs"):
            if label in self.labels:
                index = self.labels.index(label)
                self.headings[index] = np.eye(3)
                self.corrections[index] = np.eye(3)
        self._pending = None

    def record_bird_dog_gravity(self, raw_quats, raw_accels) -> None:
        """Record raw specific-force vectors for static gravity estimation."""
        if self._bird_dog_gravity is not None:
            self._bird_dog_gravity.append((np.asarray(raw_quats, dtype=np.float64).copy(),
                                           np.asarray(raw_accels, dtype=np.float64).copy()))

    def finish_bird_dog_sequence(self) -> str:
        if self._bird_dog_samples is None:
            return "bird dog calibration: no active sequence"
        result = self._solve_bird_dog_sequence()
        self._bird_dog_samples = None
        self._pending = None
        return result

    def request_pose_hold(self, pose: str, seconds: float | None = None) -> None:
        """A known hold of the lying games (KNOWN_HOLDS); the step's last `seconds` are searched."""
        if pose not in KNOWN_HOLDS:
            raise ValueError(f"unknown pose hold {pose!r}")
        search = int(round(min(max(seconds or 8.0, 1.0), HOLD_HISTORY_FRAMES / 60) * 60))
        self._pending = {"kind": "pose_hold", "pose": pose, "search": search, "frames": 0}

    def request_bird_dog_reach(self, left_side: bool) -> None:
        """Calibrate the forward arm and rearward shin in one mirrored bird-dog pose."""
        self._pending = {"kind": "bird_dog_left" if left_side else "bird_dog_right", "frames": 0}

    def advance(self, frames: int) -> str | None:
        """Call after pushing `frames` window frames; returns a summary once a request is solved."""
        if self._pending is None:
            return None
        self._pending["frames"] -= frames
        if self._pending["frames"] > 0:
            return None
        pending, self._pending = self._pending, None
        if len(self._raw) < WINDOW_FRAMES:
            return "pose calibration skipped: not enough input yet"
        measured, still = self._window_means()
        gravity = self._window_gravity(still)
        if pending["kind"] == "stand":
            return self._solve_stand(measured, still)
        if pending["kind"] == "arms":
            if self._stand is not None:
                return self._solve_two_pose_arms(measured, still)
            return self._solve_arms_forward(measured, still)
        if pending["kind"] == "pose_hold":
            return self._solve_pose_hold(pending["pose"], pending["search"])
        if pending["kind"] == "bird_dog_base":
            self._bird_dog_base = (measured, still, gravity)
            return "bird dog calibration: base pose captured"
        if pending["kind"] in ("bird_dog_left", "bird_dog_right"):
            left_side = pending["kind"] == "bird_dog_left"
            self._bird_dog_reaches[pending["kind"]] = (measured, still, gravity)
            self._bird_dog_arm_holds[pending["kind"]] = self._latest_level_hold("lw" if left_side else "rw")
            if self._bird_dog_base is None:
                return "bird dog calibration: base pose missing; restart calibration"
            side = "left arm/right leg" if left_side else "right arm/left leg"
            if len(self._bird_dog_reaches) >= 2:
                result = self._solve_bird_dog_sequence()
                self._bird_dog_samples = None
                return result
            return f"bird dog calibration: {side} pose captured"
        return self._solve_coach(pending["coach"], measured, still)

    def _solve_bird_dog_reach(self, measured, still, left_side: bool) -> str:
        if "waist" not in measured or not still.get("waist"):
            return "bird dog calibration: waist moving; keep the trunk still"
        waist_index = self.labels.index("waist")
        facing = _yaw(_best_yaw([
            self.headings[waist_index] @ measured["waist"] @ self.corrections[waist_index]
        ]))
        arm = "lw" if left_side else "rw"
        shank = "rs" if left_side else "ls"
        targets = ((arm, facing @ np.array([0.0, 0.0, 1.0])),
                   (shank, facing @ np.array([0.0, 0.0, -1.0])))
        report = []
        for label, target in targets:
            if label not in measured:
                report.append(f"{label} unavailable")
                continue
            if not still.get(label):
                report.append(f"{label} moving")
                continue
            self._align_bone_to_direction(label, measured[label], target, report)
        return "bird dog calibration: " + (", ".join(report) if report else "no usable sensors")

    def _solve_bird_dog_sequence(self) -> str:
        """Forearms: a right-multiplied tilt (2 DOF) that makes them vertical in the base hold, then a
        world heading (1 DOF) from the reach that holds each arm forward, level.
        Shanks: a minimal swing fit to the base and reach holds.

        The forearm error measured live is a world-yaw (heading) error: it is identical with the arm
        forward, palm-in or sideways. Base-pose forearms are vertical and show no heading, so the
        forearm heading comes from the reach alone; a right-multiplied swing fitted to both holds
        would split the error between them."""
        from scipy.optimize import least_squares

        base_measured, base_still, base_gravity = self._bird_dog_base
        reaches = {"bird_dog_left": self._bird_dog_reaches.get("bird_dog_left"),
                   "bird_dog_right": self._bird_dog_reaches.get("bird_dog_right")}
        if reaches["bird_dog_left"] is None or reaches["bird_dog_right"] is None:
            return "bird dog calibration incomplete: a reach is missing"
        if "waist" not in self.labels:
            return "bird dog calibration skipped: waist sensor unavailable"
        waist_index = self.labels.index("waist")

        def pose_facing(pose):
            measured, still, _ = pose
            if not still.get("waist"):
                return None
            waist = self.headings[waist_index] @ measured["waist"] @ self.corrections[waist_index]
            return _yaw(_best_yaw([waist]))

        left_facing, right_facing = pose_facing(reaches["bird_dog_left"]), pose_facing(reaches["bird_dog_right"])
        if left_facing is None or right_facing is None:
            return "bird dog calibration skipped: waist moving during a reach"

        gravity_vectors = list(base_gravity.values())
        for _, _, pose_gravity in reaches.values():
            gravity_vectors.extend(pose_gravity.values())
        if gravity_vectors:
            ground_up = np.mean(gravity_vectors, axis=0)
            ground_norm = np.linalg.norm(ground_up)
            if ground_norm < 1e-6:
                return "bird dog calibration skipped: ground normal is ambiguous"
            ground_up /= ground_norm
            spread = max(_angle_deg(_rotation_between(vector, ground_up)) for vector in gravity_vectors)
            if spread > 20.0:
                return f"bird dog calibration skipped: ground normal unstable ({spread:.0f} deg)"
            def horizontal(direction):
                projected = direction - np.dot(direction, ground_up) * ground_up
                norm = np.linalg.norm(projected)
                return projected / norm if norm > 1e-6 else direction
            left_forward = horizontal(left_facing @ np.array([0.0, 0.0, 1.0]))
            right_forward = horizontal(right_facing @ np.array([0.0, 0.0, 1.0]))
            ground_report = f"; ground normal +Y error {_angle_deg(_rotation_between(ground_up, np.array([0.0, 1.0, 0.0]))):.1f} deg"
        else:
            left_forward = left_facing @ np.array([0.0, 0.0, 1.0])
            right_forward = right_facing @ np.array([0.0, 0.0, 1.0])
            ground_report = "; gravity unavailable, world-up fallback"

        report = []
        # Forearm mount tilt from the base hold: on all fours the forearms are vertical, which fixes
        # the 2-DOF tilt of the sensor's bone axis (measured live 13-16 deg; it makes the forearm
        # direction swing when the user only pronates). Verticality does not depend on the world
        # heading, so this is solved first and the heading below uses the corrected axis.
        base_down = -ground_up if gravity_vectors else np.array([0.0, -1.0, 0.0])
        for label in ("lw", "rw"):
            if label not in self.labels:
                continue
            if label not in base_measured or not base_still.get(label):
                report.append(f"{label} base moving, tilt kept")
                continue
            index = self.labels.index(label)
            current = self.headings[index] @ base_measured[label] @ self.corrections[index] @ BONE_AXES[label]
            tilt = _angle_deg(_rotation_between(current, base_down))
            if tilt > MAX_BIRD_DOG_FOREARM_TILT_DEG:
                report.append(f"{label} base tilt {tilt:.0f} deg rejected (forearm not vertical?)")
                continue
            target_local = (self.corrections[index].T @ base_measured[label].T @
                            self.headings[index].T @ base_down)
            self.corrections[index] = self.corrections[index] @ _rotation_between(BONE_AXES[label], target_local)
            report.append(f"{label} tilt {tilt:.0f} deg")
        # Forearm headings from the latest still, level hold of the reaching arm within its step
        # (the request moment itself may catch the arm already lowered).
        # Reference "forward" = waist heading in the BASE hold. During a reach the lumbar/pelvis yaws
        # (live 2026-11-12: -19 deg on the left reach, +18 deg on the right) while the shoulders stay
        # square, so aligning each arm to the waist of its own reach turned both forearms ~18 deg inward.
        base_forward = None
        if "waist" in base_measured and base_still.get("waist"):
            base_forward = _yaw(_best_yaw([self.headings[waist_index] @ base_measured["waist"] @
                                           self.corrections[waist_index]]))
        else:
            report.append("base waist moving, arm heading uses the reach waist")
        for label, kind in (("lw", "bird_dog_left"), ("rw", "bird_dog_right")):
            if label not in self.labels:
                continue
            hold = self._bird_dog_arm_holds.get(kind)
            if hold is None:
                report.append(f"{label} never held level and still during its reach, kept")
                continue
            arm, waist = hold
            index = self.labels.index(label)
            direction = self.headings[index] @ arm @ self.corrections[index] @ BONE_AXES[label]
            forward = base_forward if base_forward is not None else                 _yaw(_best_yaw([self.headings[waist_index] @ waist @ self.corrections[waist_index]]))
            phi = _azimuth(forward @ np.array([0.0, 0.0, 1.0])) - _azimuth(direction)
            current = np.arctan2(self.headings[index][0, 2], self.headings[index][0, 0])
            self._set_heading(label, _wrap(current + phi), report, limit=MAX_BIRD_DOG_ARM_HEADING_DEG)

        # Shanks: swing-only fit, base (parallel, free common direction) and reach (pointing back).
        targets = {"rs": -left_forward, "ls": -right_forward}
        reach_by_label = {"rs": reaches["bird_dog_left"], "ls": reaches["bird_dog_right"]}
        valid = []
        for label in ("ls", "rs"):
            reach_measured, reach_still, _ = reach_by_label[label]
            if label not in base_measured or not base_still.get(label):
                report.append(f"{label} base moving/unavailable")
            elif label not in reach_measured or not reach_still.get(label):
                report.append(f"{label} reach moving/unavailable")
            else:
                valid.append((label, reach_measured[label]))
        if not valid:
            return "bird dog calibration: " + ", ".join(report) + ground_report

        base_directions = [self.headings[self.labels.index(label)] @ base_measured[label] @
                           self.corrections[self.labels.index(label)] @ BONE_AXES[label] for label, _ in valid]
        shank_mean = np.sum(base_directions, axis=0)
        shank_mean /= max(np.linalg.norm(shank_mean), 1e-8)

        # A direction observation has rank two: rotation about the observed bone
        # axis is unobservable. Optimize only the minimum swing from the existing
        # mount correction so the null-space component remains unchanged.
        def corrected_mount(label, swing):
            existing = self.corrections[self.labels.index(label)]
            axis = BONE_AXES[label] / np.linalg.norm(BONE_AXES[label])
            reference = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            first = np.cross(axis, reference)
            first /= np.linalg.norm(first)
            second = np.cross(axis, first)
            return existing @ Rotation.from_rotvec(first * swing[0] + second * swing[1]).as_matrix()

        def directions(label, reach_orientation, mount):
            index = self.labels.index(label)
            return (self.headings[index] @ base_measured[label] @ mount @ BONE_AXES[label],
                    self.headings[index] @ reach_orientation @ mount @ BONE_AXES[label])

        n = len(valid)

        def residual(parameters):
            shank_base = parameters[2 * n:] / max(np.linalg.norm(parameters[2 * n:]), 1e-8)
            errors = []
            for position, (label, reach_orientation) in enumerate(valid):
                base, reach = directions(label, reach_orientation,
                                         corrected_mount(label, parameters[2 * position:2 * position + 2]))
                errors.extend(base - shank_base)
                errors.extend(reach - targets[label])
            return np.asarray(errors)

        fit = least_squares(residual, np.r_[np.zeros(2 * n), shank_mean], max_nfev=200)
        if not fit.success or not np.isfinite(fit.cost) or not np.all(np.isfinite(fit.x)):
            return "bird dog calibration: " + ", ".join(report) + "; shank fit failed, shanks unchanged"
        shank_base = fit.x[2 * n:] / max(np.linalg.norm(fit.x[2 * n:]), 1e-8)
        candidates, errors, worst_error = {}, [], 0.0
        for position, (label, reach_orientation) in enumerate(valid):
            candidate = corrected_mount(label, fit.x[2 * position:2 * position + 2])
            change = _angle_deg(self.corrections[self.labels.index(label)].T @ candidate)
            if change > MAX_BIRD_DOG_MOUNT_STEP_DEG:
                return ("bird dog calibration: " + ", ".join(report) + f"; shanks rejected: {label} needs "
                        f"{change:.0f} deg correction (limit {MAX_BIRD_DOG_MOUNT_STEP_DEG:.0f})")
            base, reach = directions(label, reach_orientation, candidate)
            base_error = _angle_deg(_rotation_between(base, shank_base))
            reach_error = _angle_deg(_rotation_between(reach, targets[label]))
            worst_error = max(worst_error, base_error, reach_error)
            errors.append(f"{label} base {base_error:.0f} deg/reach {reach_error:.0f} deg")
            candidates[label] = candidate
        if worst_error > MAX_BIRD_DOG_FIT_ERROR_DEG:
            return ("bird dog calibration: " + ", ".join(report) + f"; shanks rejected: fit error "
                    f"{worst_error:.0f} deg (limit {MAX_BIRD_DOG_FIT_ERROR_DEG:.0f})")
        for label, candidate in candidates.items():
            self.corrections[self.labels.index(label)] = candidate
        return ("bird dog calibration: forearm tilt/heading " + ", ".join(report) + "; shank fit " +
                ", ".join(errors) + self._bird_dog_transition_error() + ground_report)

    def _latest_level_hold(self, label: str):
        """(arm, waist) mean orientations of the latest 0.5 s window, within the reach step, in which
        the arm is level and both it and the waist are still; None if there is none."""
        if not self._bird_dog_samples:
            return None
        return self._latest_hold(np.asarray(self._bird_dog_samples[-REACH_SEARCH_FRAMES:]), label, level=True)

    def _latest_hold(self, samples: np.ndarray, label: str, level: bool):
        """(bone, waist) mean orientations of the latest 0.5 s window of `samples` in which both are still
        and, if `level`, the bone is within MAX_LEVEL_ELEVATION_DEG of horizontal; None if there is none."""
        if label not in self.labels or "waist" not in self.labels or len(samples) < WINDOW_FRAMES:
            return None
        index, waist_index = self.labels.index(label), self.labels.index("waist")
        for end in range(len(samples), WINDOW_FRAMES - 1, -5):
            window = samples[end - WINDOW_FRAMES:end]
            means = []
            for s in (index, waist_index):
                rotations = Rotation.from_matrix(window[:, s])
                mean = rotations.mean()
                if np.degrees((mean.inv() * rotations).magnitude().max()) > MAX_STILL_DEG:
                    break
                means.append(mean.as_matrix())
            if len(means) < 2:
                continue
            if not level or label not in BONE_AXES:
                return means[0], means[1]
            direction = self.headings[index] @ means[0] @ self.corrections[index] @ BONE_AXES[label]
            if abs(_elevation(direction)) <= MAX_LEVEL_ELEVATION_DEG:
                return means[0], means[1]
        return None

    def _body_axes(self, waist_raw: np.ndarray):
        """Body-relative target directions from a raw waist orientation (None if the spine is vertical,
        where the head direction has no horizontal component)."""
        index = self.labels.index("waist")
        waist = self.headings[index] @ waist_raw @ self.corrections[index]
        horizontal = lambda v: np.array([v[0], 0.0, v[2]]) / max(np.hypot(v[0], v[2]), 1e-9)
        spine, chest = waist @ np.array([0.0, 1.0, 0.0]), waist @ np.array([0.0, 0.0, 1.0])
        if np.hypot(spine[0], spine[2]) < 0.3:
            return None
        head = horizontal(spine)
        return {"+h": head, "-h": -head, "+c": horizontal(chest), "-c": -horizontal(chest),
                "up": np.array([0.0, 1.0, 0.0])}

    def _solve_pose_hold(self, pose: str, search: int) -> str:
        """Lying-game calibration hold (docs/09): world headings of level bones toward body-relative
        targets, the waist roll when lying on the back, and checks that only report."""
        spec = KNOWN_HOLDS[pose]
        if "waist" not in self.labels:
            return f"pose hold calibration ({pose}): skipped, no waist sensor"
        samples = np.asarray(self._history)[-search:]
        solved, checks = [], []
        waist_index = self.labels.index("waist")
        if spec.get("waist") == ("roll", "solve"):
            hold = self._latest_hold(samples, "waist", level=False)
            if hold is None:
                solved.append("waist moving, kept")
            else:
                current = self.headings[waist_index] @ hold[1] @ self.corrections[waist_index]
                # right-multiplied turn about the spine axis that makes the left-right axis level
                theta = np.degrees(np.arctan2(current[1, 0], current[1, 2]))
                if abs(theta) > MAX_POSE_HOLD_ROLL_DEG:
                    solved.append(f"waist roll {theta:+.0f}° rejected (not lying on the back?)")
                else:
                    self.corrections[waist_index] = self.corrections[waist_index] @ _RY(theta)
                    solved.append(f"waist roll {theta:+.0f}°")
        for label, (target_name, action) in spec.items():
            if label == "waist" or label not in self.labels or label not in BONE_AXES:
                continue
            vertical_target = target_name == "up"
            hold = self._latest_hold(samples, label, level=not vertical_target)
            report = solved if action == "solve" else checks
            if hold is None:
                report.append(f"{label} never held {'still' if vertical_target else 'level and still'}")
                continue
            axes = self._body_axes(hold[1])
            if axes is None:
                report.append(f"{label}: trunk upright, no body axis")
                continue
            index = self.labels.index(label)
            direction = self.headings[index] @ hold[0] @ self.corrections[index] @ BONE_AXES[label]
            target = axes[target_name]
            if vertical_target:
                angle = np.degrees(np.arccos(np.clip(direction @ target / np.linalg.norm(direction), -1, 1)))
                checks.append(f"{label} {angle:.0f}° from vertical"
                              + ("" if angle <= POSE_HOLD_UP_CHECK_DEG else " CHECK FAILED"))
                continue
            phi = _wrap(_azimuth(target) - _azimuth(direction))
            if action == "solve":
                current = np.arctan2(self.headings[index][0, 2], self.headings[index][0, 0])
                self._set_heading(label, _wrap(current + phi), solved, limit=180.0)
            else:
                checks.append(f"{label} {np.degrees(phi):+.0f}° from {target_name}"
                              + ("" if abs(np.degrees(phi)) <= POSE_HOLD_CHECK_DEG else " CHECK FAILED"))
        text = f"pose hold calibration ({pose}): " + (", ".join(solved) if solved else "nothing solved")
        if checks:
            text += "; checks: " + ", ".join(checks)
        return text

    def _bird_dog_transition_error(self) -> str:
        """Report trajectory smoothness under one fixed mount without fitting motion as bias."""
        if not self._bird_dog_samples:
            return "; transitions unavailable"
        samples = np.asarray(self._bird_dog_samples)
        reports = []
        for label in ("lw", "rw", "ls", "rs"):
            if label not in self.labels:
                continue
            index = self.labels.index(label)
            corrected = self.headings[index] @ samples[:, index] @ self.corrections[index]
            raw_rot = Rotation.from_matrix(samples[:, index])
            corrected_rot = Rotation.from_matrix(corrected)
            raw_step = (raw_rot[:-1].inv() * raw_rot[1:]).magnitude()
            step = (corrected_rot[:-1].inv() * corrected_rot[1:]).magnitude()
            if len(step):
                invariant_error = np.degrees(np.max(np.abs(raw_step - step)))
                reports.append(f"{label} transition p95 {np.degrees(np.percentile(step, 95)):.0f} deg/frame, "
                               f"fixed-correction delta error {invariant_error:.2g} deg")
        return "; transitions " + ", ".join(reports)

    def _window_gravity(self, still: Dict[str, bool]) -> Dict[str, np.ndarray]:
        if not self._bird_dog_gravity:
            return {}
        result = {}
        for index, label in enumerate(self.labels):
            if not still.get(label, False):
                continue
            vectors = []
            for quats, accels in self._bird_dog_gravity[-WINDOW_FRAMES:]:
                accel = accels[index]
                magnitude = np.linalg.norm(accel)
                if not np.all(np.isfinite(accel)) or not 7.0 <= magnitude <= 12.5:
                    continue
                vectors.append(SERVER_TO_SMPL @ Rotation.from_quat(quats[index]).apply(accel / magnitude))
            if len(vectors) >= WINDOW_FRAMES // 3:
                mean = np.mean(vectors, axis=0)
                if np.linalg.norm(mean) > 1e-6:
                    result[label] = mean / np.linalg.norm(mean)
        return result

    def _align_bone_to_direction(self, label: str, measured: np.ndarray, target: np.ndarray,
                                 report: List[str]) -> None:
        index = self.labels.index(label)
        axis = BONE_AXES[label]
        target = target / np.linalg.norm(target)
        current = self.headings[index] @ measured @ self.corrections[index] @ axis
        angle = _angle_deg(_rotation_between(current, target))
        if angle > MAX_BIRD_DOG_HEADING_DEG:
            report.append(f"{label} {angle:.0f} deg rejected")
            return
        target_local = self.corrections[index].T @ measured.T @ self.headings[index].T @ target
        self.corrections[index] = self.corrections[index] @ _rotation_between(axis, target_local)
        report.append(f"{label} {angle:.0f} deg")

    def _make_bones_parallel(self, labels, measured, still) -> str:
        available = [label for label in labels if label in measured]
        if len(available) != len(labels):
            return "sensors unavailable"
        moving = [label for label in available if not still.get(label)]
        if moving:
            return f"{', '.join(moving)} moving"
        directions = []
        for label in available:
            index = self.labels.index(label)
            directions.append(self.headings[index] @ measured[label] @
                              self.corrections[index] @ BONE_AXES[label])
        mean = np.sum(directions, axis=0)
        mean_norm = np.linalg.norm(mean)
        if mean_norm < 1e-5:
            return "directions oppose; check the pose"
        target = mean / mean_norm
        report = []
        for label in available:
            self._align_bone_to_direction(label, measured[label], target, report)
        return ", ".join(report)

    def _window_means(self):
        window = np.asarray(self._raw)[-WINDOW_FRAMES:]          # (T, S, 3, 3)
        measured, still = {}, {}
        for s, label in enumerate(self.labels):
            rotations = Rotation.from_matrix(window[:, s])
            mean = rotations.mean()
            measured[label] = mean.as_matrix()
            still[label] = float(np.degrees((mean.inv() * rotations).magnitude().max())) <= MAX_STILL_DEG
        return measured, still

    def _bird_dog_hold_gravity(self, label: str, frame_range: tuple[int, int]) -> np.ndarray | None:
        if not self._bird_dog_gravity or label not in self.labels:
            return None
        start, end = frame_range
        samples = self._bird_dog_gravity[max(0, start):min(end, len(self._bird_dog_gravity))]
        if len(samples) < WINDOW_FRAMES // 2:
            return None
        index = self.labels.index(label)
        world_up = []
        for quats, accels in samples:
            q, accel = quats[index], accels[index]
            if not np.all(np.isfinite(accel)) or not 7.0 <= np.linalg.norm(accel) <= 12.5:
                continue
            measured_up = SERVER_TO_SMPL @ Rotation.from_quat(q).apply(accel / np.linalg.norm(accel))
            world_up.append(measured_up)
        if len(world_up) < WINDOW_FRAMES // 3:
            return None
        mean = np.mean(world_up, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-6 else None

    def _set_heading(self, label: str, phi: float, report: List[str], limit: float = MAX_HEADING_DEG) -> None:
        if abs(np.degrees(phi)) > limit:
            report.append(f"{label} {np.degrees(phi):+.0f}° rejected")
            return
        self.headings[self.labels.index(label)] = _yaw(phi)
        report.append(f"{label} {np.degrees(phi):+.0f}°")

    def _solve_stand(self, measured, still) -> str:
        """Waist and shanks are vertical when standing straight: a full body-frame correction makes them
        exactly upright (tilt and twist; heading stays with the yaw reset). Forearms are kept for pose B."""
        if "waist" not in measured or not still.get("waist"):
            return "stand calibration skipped: waist moving"
        self._stand = {"measured": measured, "still": still}
        report = []
        for label in ("waist", "ls", "rs"):
            if label not in measured:
                continue
            if not still[label]:
                report.append(f"{label} moving")
                continue
            # keep the sensor's own heading (only tilt/twist is known here), make it otherwise upright
            facing = _yaw(_best_yaw([measured[label]]))
            correction = measured[label].T @ facing
            angle = _angle_deg(correction)
            if angle > MAX_STAND_TILT_DEG:
                report.append(f"{label} {angle:.0f}° rejected")
                continue
            index = self.labels.index(label)
            self.corrections[index] = correction
            self.headings[index] = np.eye(3)
            report.append(f"{label} {angle:.0f}°")
        return "stand calibration: " + ", ".join(report)

    def _solve_two_pose_arms(self, measured, still) -> str:
        """Forearms seen in two known poses: solve a world heading H (1 DOF) and a body-frame mount C
        (3 DOF) so that H R_meas C equals the known pose in both, facing the waist's heading."""
        from scipy.optimize import least_squares
        stand = self._stand
        facing = _yaw(_best_yaw([stand["measured"]["waist"]]))
        report = []
        for label in ("lw", "rw"):
            if label not in measured:
                continue
            if not (still[label] and stand["still"].get(label)):
                report.append(f"{label} moving")
                continue
            poses = [(stand["measured"][label], facing @ ARMS_DOWN[label]),
                     (measured[label], facing @ ARMS_FORWARD[label])]

            def residual(x):
                h, c = _yaw(x[0]), Rotation.from_rotvec(x[1:]).as_matrix()
                return np.concatenate([Rotation.from_matrix((h @ m @ c).T @ t).as_rotvec() for m, t in poses])

            best = min((least_squares(residual, np.r_[h0, Rotation.from_matrix(poses[0][0].T @ _yaw(-h0) @ poses[0][1]).as_rotvec()])
                        for h0 in np.radians([-90.0, -45.0, 0.0, 45.0, 90.0])), key=lambda r: r.cost)
            fit = np.degrees(np.sqrt(np.mean(np.sum(best.fun.reshape(2, 3) ** 2, axis=1))))
            heading, mount = np.degrees(_wrap(best.x[0])), np.degrees(np.linalg.norm(best.x[1:]))
            if fit > MAX_TWO_POSE_FIT_DEG:
                report.append(f"{label} poses inconsistent ({fit:.0f}°), kept")
                continue
            index = self.labels.index(label)
            self.headings[index] = _yaw(best.x[0])
            self.corrections[index] = Rotation.from_rotvec(best.x[1:]).as_matrix()
            report.append(f"{label} heading {heading:+.0f}° mount {mount:.0f}° (fit {fit:.0f}°)")
        return "arm calibration (2 poses): " + ", ".join(report)

    def _solve_arms_forward(self, measured, still) -> str:
        if "waist" not in measured:
            return "arm calibration skipped: no waist sensor"
        forward = measured["waist"] @ np.array([0.0, 0.0, 1.0])   # chest normal = facing while standing
        report = []
        for label in ("lw", "rw"):
            if label not in measured:
                continue
            if not still[label]:
                report.append(f"{label} moving")
                continue
            direction = self.headings[self.labels.index(label)] @ measured[label] @ BONE_AXES[label]
            if abs(_elevation(direction)) > MAX_LEVEL_ELEVATION_DEG:
                report.append(f"{label} not level ({_elevation(direction):+.0f}°)")
                continue
            # compose with the heading already applied, which the direction above includes
            phi = _azimuth(forward) - _azimuth(direction)
            current = np.arctan2(self.headings[self.labels.index(label)][0, 2], self.headings[self.labels.index(label)][0, 0])
            self._set_heading(label, _wrap(current + phi), report)
        return "arm calibration: " + ", ".join(report)

    def _make_shanks_parallel(self, measured, still, report: List[str]) -> None:
        """In the start poses (kneeling, table-top, bridge) the shanks are parallel. Both are turned onto
        their mean heading, keeping the common direction from the yaw reset and removing the symmetric
        inward/outward error measured live (the coach retarget is not used: it disagreed in sign with the
        coach's own capture)."""
        if "ls" not in measured or "rs" not in measured:
            return
        if not (still.get("ls") and still.get("rs")):
            report.append("shanks moving")
            return
        # include the stand-pose tilt correction, then choose the heading that makes the shanks parallel
        left = measured["ls"] @ self.corrections[self.labels.index("ls")] @ BONE_AXES["ls"]
        right = measured["rs"] @ self.corrections[self.labels.index("rs")] @ BONE_AXES["rs"]
        if max(abs(_elevation(left)), abs(_elevation(right))) > MAX_LEVEL_ELEVATION_DEG:
            report.append("shanks vertical, kept")
            return
        mean = np.arctan2(np.sin(_azimuth(left)) + np.sin(_azimuth(right)),
                          np.cos(_azimuth(left)) + np.cos(_azimuth(right)))
        for label, direction in (("ls", left), ("rs", right)):
            self._set_heading(label, _wrap(mean - _azimuth(direction)), report, limit=MAX_PARALLEL_HEADING_DEG)
        report.append("(shanks made parallel)")

    def _solve_coach(self, coach, measured, still) -> str:
        usable = [label for label in coach if still.get(label)]
        if not usable:
            return "pose calibration skipped: sensors moving (hold the start pose)"
        # The user's facing relative to the coach: the waist defines body heading when it is usable,
        # otherwise the yaw maximising sum tr(Y^T R_meas R_coach^T) over all usable sensors.
        reference = ["waist"] if "waist" in usable else usable
        yaw = _yaw(_best_yaw([measured[label] @ coach[label].T for label in reference]))
        report = []
        if self.mode == "heading":
            self._make_shanks_parallel(measured, still, report)
        for label in coach:
            if not still.get(label):
                report.append(f"{label} moving")
                continue
            if self.mode == "heading":
                if label in ("waist", "ls", "rs") or label not in BONE_AXES:
                    continue    # the waist is the heading reference; shanks were handled above
                # Heading is only observable for a bone that is far from vertical, and only its pointing
                # direction is compared: twist (e.g. palm orientation) differs between people.
                user_dir = measured[label] @ BONE_AXES[label]
                coach_dir = yaw @ coach[label] @ BONE_AXES[label]
                if max(abs(_elevation(user_dir)), abs(_elevation(coach_dir))) > MAX_LEVEL_ELEVATION_DEG:
                    report.append(f"{label} vertical, kept")
                    continue
                self._set_heading(label, _wrap(_azimuth(coach_dir) - _azimuth(user_dir)), report,
                                  limit=MAX_POSE_HEADING_DEG)
                continue
            correction = measured[label].T @ yaw @ coach[label]
            angle = _angle_deg(correction)
            if angle > MAX_CORRECTION_DEG:
                report.append(f"{label} {angle:.0f}° rejected")
                continue
            self.corrections[self.labels.index(label)] = correction
            report.append(f"{label} {angle:.0f}°")
        return f"pose calibration ({self.mode}): " + ", ".join(report)

    def measured_frame(self, lag_frames: int):
        """Corrected (n_labels, 3, 3) input of the frame an output lagging `lag_frames` represents."""
        return self._corrected[-1 - lag_frames] if len(self._corrected) > lag_frames else None

    def anchor(self, local: torch.Tensor, lag_frames: int) -> torch.Tensor:
        """Replace sensed bones' global orientation in a (24, 3, 3) local pose with the measurement."""
        if not self.anchor_enabled or len(self._corrected) <= lag_frames:
            return local
        measured = self._corrected[-1 - lag_frames]
        overrides = {LABEL_JOINT[label]: torch.from_numpy(measured[s]).to(local)
                     for s, label in enumerate(self.labels) if label in LABEL_JOINT}
        result = local.clone()
        global_rot: List[torch.Tensor] = []
        for joint, parent in enumerate(SMPL_PARENTS):
            parent_rot = global_rot[parent] if parent >= 0 else torch.eye(3, dtype=local.dtype)
            if joint in overrides:
                result[joint] = parent_rot.transpose(-1, -2) @ overrides[joint]
                global_rot.append(overrides[joint])
            else:
                global_rot.append(parent_rot @ local[joint])
        return result
