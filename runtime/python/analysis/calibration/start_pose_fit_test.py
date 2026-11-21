"""python -m unittest analysis.calibration.start_pose_fit_test

Synthetic: true start pose = template; measured = world yaw drift @ true @ small mount error.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from production import start_pose_fit as spf
from production.pose_calibration import BONE_AXES, PoseCalibrator, _yaw
from production.reach_heading import HOLD_FRAMES, ReachHeading

LABELS = ["lw", "rw", "ls", "rs", "waist"]
FACING = _yaw(np.radians(37))
# four-point trunk: spine (+y) to the head (+z), chest (+z) down
WAIST = FACING @ Rotation.from_euler("x", 90, degrees=True).as_matrix()


# lying on the back: spine (+y) to the head (+z), chest (+z) up, body-left toward -x
SUPINE = FACING @ np.array([[-1.0, 0, 0], [0, 0, 1], [0, 1, 0]])
# side lying, head +z: left side down (body-left -y, chest -x) / right side down (body-left +y, chest +x)
SIDE_LEFT = FACING @ np.array([[0.0, 0, -1], [-1, 0, 0], [0, 1, 0]])
SIDE_RIGHT = FACING @ np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
WAISTS = {"quadruped": WAIST, "supine": SUPINE, "tabletop": SUPINE}


class StartPoseFit(unittest.TestCase):
    def test_every_template_recovers_its_start_pose(self):
        drift = {"lw": -97, "rw": -150, "ls": 5, "rs": -5}
        mount = Rotation.from_euler("x", 15, degrees=True).as_matrix()
        for (action, start_pose), template in spf.TEMPLATES.items():
            with self.subTest(action=action):
                waist = (SIDE_LEFT if action == "sidebridge_left" else SIDE_RIGHT if action == "sidebridge_right"
                         else WAISTS[start_pose])
                holds = {label: _yaw(np.radians(drift[label])) @ FACING @ bone @ mount
                         for label, bone in template.items()}
                fitted, rejected = spf.fit(template, holds, waist)
                self.assertEqual(rejected, [])
                for label, (heading, correction, *_rest) in fitted.items():
                    np.testing.assert_allclose(heading @ holds[label] @ correction, FACING @ template[label],
                                               atol=1e-9)

    def test_side_bridge_rejects_lying_on_the_other_side(self):
        right_down = {label: FACING @ bone for label, bone in spf.TEMPLATES[("sidebridge_right", "side_rest")].items()}
        rejected = spf.fit(spf.TEMPLATES[("sidebridge_left", "side_rest")], right_down, SIDE_RIGHT)[1]
        self.assertTrue({"ls", "rs"} <= set(rejected), rejected)

    def test_crunch_rejects_arms_at_the_sides(self):
        # Hands behind the head: palm up against the head, wrist below the raised elbow.  Arms at the sides:
        # palm down, level.  Both heading-free (gravity), so the crunch template must refuse the latter
        # instead of turning it ~180 deg (live crunch 2026-11-19: hands then reconstructed in front of the face).
        sides = {label: _yaw(np.radians(55 * (1 if label == "lw" else -1))) @ FACING @ bone
                 for label, bone in spf.TEMPLATES[("hipthrust", "supine")].items()}    # forearms angled out
        rejected = spf.fit(spf.TEMPLATES[("crunch", "supine")], sides, SUPINE)[1]
        self.assertTrue({"lw", "rw"} <= set(rejected), rejected)

    def test_crunch_hands_behind_head_keeps_an_undrifted_heading(self):
        # Live hold crunch_20261119_232340 3-15 s, raw (undrifted) body-frame forearm frames.
        measured = {"lw": np.array([[0.76, -0.63, 0.15], [-0.60, -0.77, -0.22], [0.25, 0.08, -0.97]]),
                    "rw": np.array([[0.67, 0.72, -0.17], [0.68, -0.69, -0.25], [-0.30, 0.05, -0.95]])}
        template = spf.TEMPLATES[("crunch", "supine")]
        holds = {label: FACING @ bone for label, bone in template.items()}
        holds.update({label: FACING @ Rotation.from_matrix(m).as_matrix() for label, m in measured.items()})
        fitted, rejected = spf.fit(template, holds, SUPINE)
        self.assertEqual(rejected, [])
        for label in ("lw", "rw"):
            self.assertLess(abs(fitted[label][2]), 10.0, label)    # heading change (deg)

    def test_palm_turned_in_is_accepted_for_arms_at_the_sides(self):
        template = spf.TEMPLATES[("hipthrust", "supine")]
        holds = {label: FACING @ bone for label, bone in template.items()}
        # right palm turned 45 deg about the forearm (live hip thrust 2026-11-19)
        holds["rw"] = holds["rw"] @ Rotation.from_rotvec(np.radians(45) * BONE_AXES["rw"]).as_matrix()
        fitted, rejected = spf.fit(template, holds, SUPINE)
        self.assertEqual(rejected, [])
        self.assertAlmostEqual(fitted["rw"][2], 0.0, places=6)    # heading untouched by the twist

    def test_pose_gate_rejects_a_raised_forearm(self):
        template = spf.TEMPLATES[("birddog", "quadruped")]
        holds = {label: FACING @ bone for label, bone in template.items()}
        holds["lw"] = FACING @ Rotation.from_euler("z", -40, degrees=True).as_matrix()   # 40 deg below level
        self.assertFalse(spf.pose_matches(template, holds))
        self.assertIn("lw", spf.fit(template, holds, WAIST)[1])


class ReachAlignment(unittest.TestCase):
    def test_held_level_reach_turns_heading_to_the_head(self):
        cal = PoseCalibrator(LABELS)
        reach = ReachHeading(LABELS)
        reach.activate("birddog")
        # left forearm level, pointing 80 deg to the side of the head direction
        head = WAIST @ np.array([0.0, 1.0, 0.0])
        side = _yaw(np.radians(80)) @ head
        frame = np.tile(np.eye(3), (5, 1, 1))
        frame[0] = Rotation.align_vectors([side], [BONE_AXES["lw"]])[0].as_matrix()
        frame[4] = WAIST
        for _ in range(HOLD_FRAMES + 60):
            reach.update(cal, frame)
        direction = cal.headings[0] @ frame[0] @ cal.corrections[0] @ BONE_AXES["lw"]
        self.assertGreater(direction @ head, np.cos(np.radians(1)))
        self.assertEqual([e["label"] for e in reach.events].count("lw"), 1)   # one turn, then settled

    def test_arm_laid_toward_the_feet_after_a_reach_is_ignored(self):
        cal = PoseCalibrator(LABELS)
        reach = ReachHeading(LABELS)
        reach.activate("deadbug")
        head = SUPINE @ np.array([0.0, 1.0, 0.0])
        frame = np.tile(np.eye(3), (5, 1, 1))
        frame[4] = SUPINE
        frame[1] = Rotation.align_vectors([[0.0, 1.0, 0.0]], [BONE_AXES["rw"]])[0].as_matrix()   # right arm up
        # left forearm reaches overhead, 30 deg off the head direction, then is laid toward the feet
        for heading_deg, frames in ((30, HOLD_FRAMES + 40), (180, HOLD_FRAMES + 40)):
            frame[0] = Rotation.align_vectors([_yaw(np.radians(heading_deg)) @ head], [BONE_AXES["lw"]])[0].as_matrix()
            for _ in range(frames):
                reach.update(cal, frame)
        applied = [e for e in reach.events if e["label"] == "lw" and not e.get("ignored")]
        self.assertEqual(len(applied), 1)
        self.assertAlmostEqual(applied[0]["errorDeg"], -30.0, delta=1.0)
        self.assertTrue(any(e.get("ignored") for e in reach.events if e["label"] == "lw"))

    def test_other_actions_untouched(self):
        cal = PoseCalibrator(LABELS)
        reach = ReachHeading(LABELS)
        reach.activate("deadbug")
        frame = np.tile(np.eye(3), (5, 1, 1))
        frame[4] = WAIST
        for _ in range(HOLD_FRAMES + 10):
            reach.update(cal, frame)
        np.testing.assert_allclose(cal.headings, np.tile(np.eye(3), (5, 1, 1)))


if __name__ == "__main__":
    unittest.main()
