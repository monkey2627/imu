"""Action-specific hand/shank calibration, separate from the unchanged bird-dog solver.

Only a bone's direction is observed: keep axial twist, solve a world heading
and a bounded local swing. All steps must pass before a new set is committed.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from production.bridge_contact import (BRIDGE_MODELS, TRUNK_BIAS_MAX_SPREAD_DEG, WAIST_FLAT_MAX_DEG, trunk_bias,
                                       waist_flat_fit)
from production.pose_calibration import (
    BONE_AXES, _angle_deg, _azimuth,
    _elevation, _rotation_between, _wrap, _yaw,
)


def step(pose, instruction, seconds=8):
    return {"pose": pose, "instruction": instruction, "seconds": seconds}


STRAIGHT = step("supine_straight", "仰卧，双腿伸直、平行放在垫上；双臂伸直贴身体两侧，手指朝脚、掌心向下。保持不动。", 10)
ARMS = step("supine_arms", "仰卧屈膝、脚踩地；双臂伸直贴身体两侧，手指朝脚、掌心向下。保持不动。", 10)
TABLETOP = step("tabletop", "仰卧，大腿竖直、双膝约90度，小腿水平指向脚端；双臂伸直朝天，保持不动。", 10)

EXERCISES = {
    "deadbug": {"label": "死虫", "steps": [ARMS, TABLETOP,
        step("deadbug_left", "左臂沿头顶方向伸直、靠近水平，右腿向脚端伸直；腰背稳定，保持不动。"),
        step("deadbug_right", "先回到桌面式，再换右臂向头顶伸直、左腿向脚端伸直；保持不动。"),
        step("tabletop_check", "回到死虫起始姿势：双臂朝天，大腿竖直、小腿水平，保持不动。")]},
    # Glute bridge: the first hold also levels the waist (lie flat: back, buttocks and head on the floor), the
    # second records the network's trunk output while lying flat (bridge models only, bridge_contact.py).
    "hipthrust": {"label": "臀桥", "steps": [
        step("supine_straight_flat", "仰卧，背部、臀部和头都平贴垫子；双腿伸直、平行放在垫上；双臂伸直贴身体两侧，"
                                      "手指朝脚、掌心向下。保持不动。", 10),
        step("supine_rest_trunk", "屈膝、双脚踩地与髋同宽；背部、臀部和头继续平贴垫子，手臂平放体侧，保持不动。"),
        step("bridge_check", "缓慢抬臀到臀桥姿势；双臂仍平放地面、手指朝脚，双脚不移动，保持不动。"),
        step("supine_check", "缓慢落臀回到准备姿势，双臂仍贴体侧，保持不动。", 6)]},
    "sidebridge_left": {"label": "侧桥（左侧支撑）", "steps": [STRAIGHT,
        step("side_left", "左侧卧，左肘在肩下、左前臂水平朝胸前；右臂伸直沿身体指向脚端。先不抬髋，保持不动。", 10),
        step("side_left_check", "维持左肘支撑，缓慢抬髋；左前臂朝胸前、右臂沿体侧，保持不动。"),
        step("side_legs_check", "落髋后保持侧卧，双腿伸直、双小腿沿身体指向脚端且平行，保持不动。")]},
    "sidebridge_right": {"label": "侧桥（右侧支撑）", "steps": [STRAIGHT,
        step("side_right", "右侧卧，右肘在肩下、右前臂水平朝胸前；左臂伸直沿身体指向脚端。先不抬髋，保持不动。", 10),
        step("side_right_check", "维持右肘支撑，缓慢抬髋；右前臂朝胸前、左臂沿体侧，保持不动。"),
        step("side_legs_check", "落髋后保持侧卧，双腿伸直、双小腿沿身体指向脚端且平行，保持不动。")]},
    "crunch": {"label": "卷腹", "steps": [STRAIGHT,
        step("supine_check", "屈膝、脚踩地，双臂伸直平放体侧，先放松躺好，保持不动。"),
        step("crunch_check", "小幅卷腹抬起肩胛，双臂伸直沿大腿朝脚端伸出、尽量平行地面；不拉头，保持不动。"),
        step("supine_check", "缓慢躺回垫上，双臂放回体侧，保持不动。", 6)]},
    "lyingcat": {"label": "卧位猫（四点跪姿猫式）", "steps": [
        step("cat_neutral", "四点跪姿中立位：双手固定在肩下、双膝固定在髋下，前臂竖直，小腿沿身体向后。背部自然平直，保持不动。", 4)]},
}

# Target directions are expressed in the waist frame projected to the floor.
# "fit" corrects heading + inclination; "check" never changes a correction.
POSES = {
    "supine_straight": {label: ("-h", "fit") for label in BONE_AXES},
    "supine_arms": {label: ("-h", "fit") for label in ("lw", "rw")},
    "tabletop": {"ls": ("-h", "fit"), "rs": ("-h", "fit"), "lw": ("up", "check"), "rw": ("up", "check")},
    "deadbug_left": {"lw": ("+h", "check"), "rs": ("-h", "check")},
    "deadbug_right": {"rw": ("+h", "check"), "ls": ("-h", "check")},
    "tabletop_check": {"ls": ("-h", "check"), "rs": ("-h", "check"), "lw": ("up", "check"), "rw": ("up", "check")},
    "supine_check": {"lw": ("-h", "check"), "rw": ("-h", "check")},
    "supine_straight_flat": {label: ("-h", "fit") for label in BONE_AXES},          # + waist lie-flat fit
    "supine_rest_trunk": {"lw": ("-h", "check"), "rw": ("-h", "check")},            # + trunk output bias
    "bridge_check": {"lw": ("-h", "check"), "rw": ("-h", "check")},
    "crunch_check": {"lw": ("-h", "check"), "rw": ("-h", "check")},
    "side_left": {"lw": ("+c", "check"), "rw": ("-h", "check")},
    "side_right": {"rw": ("+c", "check"), "lw": ("-h", "check")},
    "side_legs_check": {"ls": ("-h", "check"), "rs": ("-h", "check")},
    "cat_shanks": {"ls": ("-h", "fit"), "rs": ("-h", "fit")},
    "cat_left": {"lw": ("+h", "fit")},
    "cat_right": {"rw": ("+h", "fit")},
    "cat_check": {"lw": ("down", "check"), "rw": ("down", "check"), "ls": ("-h", "check"), "rs": ("-h", "check")},
    "cat_neutral": {"lw": ("down", "fit"), "rw": ("down", "fit"),
                    "ls": ("-h", "fit"), "rs": ("-h", "fit")},
}
POSES["side_left_check"] = POSES["side_left"]
POSES["side_right_check"] = POSES["side_right"]
MAX_TILT_DEG = 30.0
CHECK_DEG = 25.0
CAT_WRIST_CHECK_DEG = 40.0
PREPARE_SECONDS = 4


class ExerciseCalibration:
    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.action = None
        self.index = 0
        self.elapsed = 0
        self.active = False
        self.results = []
        self.backup = None
        self.outcome = "idle"
        self.reference_waist = None
        self.outputs = deque(maxlen=600)      # network outputs (bridge models) during the current hold

    def observe_output(self, local, model_id):
        """Raw network output (24,3,3) after each inference; kept only for bridge models while calibrating."""
        if self.active and model_id in BRIDGE_MODELS:
            self.outputs.append(np.asarray(local, dtype=np.float64).copy())

    def start(self, action):
        if action not in EXERCISES:
            raise ValueError(f"unknown exercise {action!r}")
        if not self.calibrator.calibration_enabled:
            raise ValueError("请先开启姿势校准")
        missing = set(BONE_AXES) | {"waist"}
        missing -= set(self.calibrator.labels)
        if missing:
            raise ValueError(f"missing sensors: {', '.join(sorted(missing))}")
        self.cancel()
        cal = self.calibrator
        if cal._bird_dog_samples is not None or cal._pending is not None:
            raise ValueError("已有校准进行中，请先完成当前校准")
        self.backup = (cal.headings.copy(), cal.corrections.copy(), cal.bridge_waist, cal.bridge_trunk)
        if action == "hipthrust":     # the lie-flat calibration is re-done from scratch
            cal.bridge_waist, cal.bridge_trunk = None, None
        # Corrections belong to a wearing and an action. Never fit on top of another action.
        for label in BONE_AXES:
            i = cal.labels.index(label)
            cal.headings[i] = np.eye(3)
            cal.corrections[i] = np.eye(3)
        cal._history.clear()
        self.action, self.index, self.elapsed = action, 0, 0
        self.active, self.results, self.outcome = True, [], "running"
        self.reference_waist = None
        self.outputs.clear()

    def cancel(self):
        if self.active and self.backup is not None:
            cal = self.calibrator
            cal.headings[:], cal.corrections[:], cal.bridge_waist, cal.bridge_trunk = self.backup
            self.outcome = "cancelled"
        self.active = False
        self.backup = None

    def status(self):
        current = EXERCISES[self.action]["steps"][self.index] if self.action else None
        preparing = self.active and self.elapsed < PREPARE_SECONDS * 60
        countdown = PREPARE_SECONDS if preparing else (current["seconds"] + PREPARE_SECONDS if current else 0)
        return {"action": self.action, "active": self.active, "outcome": self.outcome,
                "step": self.index + 1, "total": len(EXERCISES[self.action]["steps"]) if self.action else 0,
                "instruction": current["instruction"] if current else "选择动作后开始校准",
                "preparing": preparing,
                "remaining": max(0, math.ceil(countdown - self.elapsed / 60)) if self.active else 0,
                "results": self.results}

    def advance(self, frames):
        """Return the captured step for the input recorder (once per hold)."""
        if not self.active:
            return None
        self.elapsed += frames
        current = EXERCISES[self.action]["steps"][self.index]
        if self.elapsed < (PREPARE_SECONDS + current["seconds"]) * 60:
            return None
        # Constrain the search to this hold, excluding the transition/preparation period.
        passed, message = self.capture(current["pose"], current["seconds"])
        self.results.append(message)
        event = {"pose": current["pose"], "seconds": current["seconds"], "message": message}
        if not passed:
            self.cancel()
            self.outcome = "failed"
        elif self.index + 1 == len(EXERCISES[self.action]["steps"]):
            self.active, self.backup, self.outcome = False, None, "complete"
        else:
            self.index += 1
            self.elapsed = 0
            self.calibrator._history.clear()
            self.outputs.clear()
        return event

    def capture(self, pose, seconds):
        """Used by both the live sequence and offline replay. Commit each hold atomically."""
        spec = POSES[pose]
        cal = self.calibrator
        samples = np.asarray(cal._history)[-int(seconds * 60):]
        old = (cal.headings.copy(), cal.corrections.copy(), cal.bridge_waist, cal.bridge_trunk)
        report, passed = [], True
        waist_hold = cal._latest_hold(samples, "waist", level=False)
        if waist_hold is None:
            return False, f"exercise calibration ({pose}): 腰部未保持稳定，请重新校准"
        waist_index = cal.labels.index("waist")
        waist = cal.headings[waist_index] @ waist_hold[1] @ cal.corrections[waist_index]
        chest_up = waist[1, 2]
        if (((pose.startswith("supine") or pose.startswith("tabletop")) and chest_up < .65)
                or (pose.startswith("cat_") and chest_up > -.65)
                or (pose.startswith("side_") and abs(chest_up) > .65)):
            return False, f"exercise calibration ({pose}): 躯干姿势不符，请按提示仰卧或四点跪姿"
        # As with bird-dog, use the neutral trunk for reach headings: the lumbar
        # sensor may yaw while an arm reaches, even if the shoulders stay square.
        reference = self.reference_waist if pose in ("cat_left", "cat_right") else None
        for label, (target_name, operation) in spec.items():
            vertical = target_name in ("up", "down")
            # Search on raw orientation: the sequence starts with identity limb corrections.
            hold = cal._latest_hold(samples, label, level=not vertical)
            if hold is None:
                report.append(f"{label}: 未找到稳定{'、接近水平' if not vertical else ''}的0.5秒")
                passed = False
                continue
            axes = cal._body_axes(reference if reference is not None else hold[1])
            if axes is None or (target_name == "+c" and np.linalg.norm(axes["+c"]) < .9):
                report.append(f"{label}: 躯干姿势不符，无法确定参考方向")
                passed = False
                continue
            target = -axes["up"] if target_name == "down" else axes[target_name]
            i = cal.labels.index(label)
            direction = cal.headings[i] @ hold[0] @ cal.corrections[i] @ BONE_AXES[label]
            if operation == "fit":
                tilt = abs(_elevation(direction))
                # A vertical fixed forearm has no useful horizontal heading, but its
                # direction is fully observable and can be fitted directly.
                if not vertical and tilt > MAX_TILT_DEG:
                    report.append(f"{label}: 倾斜{tilt:.0f}°过大，请按提示摆好姿势")
                    passed = False
                    continue
                heading = _yaw(_wrap(_azimuth(target) - _azimuth(direction))) @ cal.headings[i]
                # Resolve the remaining inclination with the minimum local swing, preserving twist.
                local_target = hold[0].T @ heading.T @ target
                mount = _rotation_between(cal.corrections[i] @ BONE_AXES[label], local_target) @ cal.corrections[i]
                cal.headings[i], cal.corrections[i] = heading, mount
                report.append(f"{label}: 朝向/倾斜已校准（倾斜{tilt:.0f}°）")
            else:
                error = _angle_deg(_rotation_between(direction, target))
                limit = CAT_WRIST_CHECK_DEG if pose == "cat_check" and label in ("lw", "rw") else CHECK_DEG
                report.append(f"{label}: 复核{error:.0f}°" + (" CHECK FAILED" if error > limit else ""))
                passed &= error <= limit
        if passed and pose == "supine_straight_flat":
            correction, tilt = waist_flat_fit(waist)
            if correction is None or tilt > WAIST_FLAT_MAX_DEG:
                report.append(f"waist: 平躺校准失败（倾斜{tilt:.0f}°），请背部平贴垫子")
                passed = False
            else:
                cal.bridge_waist = correction
                report.append(f"waist: 平躺已校准（{tilt:.0f}°）")
        if passed and pose == "supine_rest_trunk":
            outputs = list(self.outputs)[-int(seconds * 60 / 2):]
            if len(outputs) < 30:
                report.append("躯干: 未使用臀桥模型，跳过躯干平躺校准")
            else:
                bias, spread = trunk_bias(np.stack(outputs))
                if spread > TRUNK_BIAS_MAX_SPREAD_DEG:
                    report.append(f"躯干: 未保持稳定（{spread:.0f}°），请背部平贴垫子不动")
                    passed = False
                else:
                    cal.bridge_trunk = bias
                    report.append("躯干: 平躺偏差已校准（%s）" % "/".join(
                        f"{math.degrees(math.acos(np.clip((np.trace(r) - 1) / 2, -1, 1))):.0f}°" for r in bias.values()))
        if not passed:
            cal.headings[:], cal.corrections[:], cal.bridge_waist, cal.bridge_trunk = old
        elif pose == "cat_shanks":
            self.reference_waist = waist_hold[1].copy()
        return passed, f"exercise calibration ({pose}): " + ", ".join(report)
