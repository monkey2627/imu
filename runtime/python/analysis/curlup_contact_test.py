"""python -m analysis.curlup_contact_test"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from production.arm_prior import PARENTS, _globals
from production.bridge_contact import PELVIS_BACK_DEPTH
from production.curlup_contact import ARM_MAX_SPEED_DEG_S, FOREARM_CURL_MAX_DEG, CurlUpContact

SUPINE = Rotation.from_euler("x", -90, degrees=True).as_matrix()


class CurlUpContactTest(unittest.TestCase):
    def test_pelvis_level_on_the_mat_and_shoulders_follow_the_waist_sensor(self):
        # The network's pelvis rocks and its spine is ignored lying on the back: pelvis level on the mat,
        # spine3 at CURL_GAIN x the waist sensor's rise above rest (the shoulders are not forced down).
        solver = CurlUpContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE @ Rotation.from_euler("x", -12, degrees=True).as_matrix()    # pelvis rocked head-down
        local[3] = Rotation.from_euler("x", 25, degrees=True).as_matrix()
        for _ in range(20):
            solver.apply(local.copy(), 1 / 30, waist=SUPINE)
        rising = SUPINE @ Rotation.from_euler("x", 10, degrees=True).as_matrix()
        out, diag = solver.apply(local.copy(), 1 / 30, waist=rising)
        g = _globals(out)
        self.assertAlmostEqual(np.degrees(np.arcsin(g[0][1, 1])), 0.0, delta=0.5)         # pelvis level
        self.assertAlmostEqual(np.degrees(np.arcsin(g[9][1, 1])), 18.0, delta=1.0)        # 1.8 x 10 deg
        self.assertAlmostEqual(diag["pelvisHeightM"], PELVIS_BACK_DEPTH, delta=0.002)

    def test_upper_arm_jump_is_limited_and_forearm_is_preserved(self):
        solver = CurlUpContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE
        solver.apply(local, 1 / 30)
        jumped = local.copy()
        # forearm swung toward the feet: not a hands-behind-head arm, so the sensed forearm is kept
        jumped[16] = Rotation.from_euler("z", -60, degrees=True).as_matrix()
        before = _globals(jumped)[18]
        out, diag = solver.apply(jumped, 1 / 30)
        after = _globals(out)[18]
        self.assertLessEqual(diag["arms"]["left"]["appliedDeltaDeg"], ARM_MAX_SPEED_DEG_S / 30 + 0.1)
        self.assertTrue(np.allclose(before, after, atol=1e-8))

    @staticmethod
    def _left_forearm(local, direction_body):
        """Set the left elbow so the forearm (+x bone) points along `direction_body` (SMPL body frame)."""
        g = _globals(local)
        world = SUPINE @ np.asarray(direction_body, float)
        forearm = Rotation.align_vectors([world], [[1.0, 0.0, 0.0]])[0].as_matrix()
        local[18] = g[16].T @ forearm
        return forearm

    def _wrist_head(self, solver, local):
        g = _globals(local)
        p = np.zeros((24, 3))
        for j in range(1, 24):
            p[j] = p[PARENTS[j]] + g[PARENTS[j]] @ solver.contacts.offsets[j]
        return float(np.linalg.norm(p[20] - p[15]))

    def test_hand_behind_head_pulls_the_wrist_to_the_head_near_the_sensed_forearm(self):
        # Live hold crunch_20261119_232340: left forearm toward the centre, slightly headward, wrist
        # below the raised elbow (body frame: -x centre, +y head, -z back).  Upper arm left in T-pose.
        # Two-bone solve: the wrist goes behind the head, the forearm stays within 45 deg of the sensor.
        solver = CurlUpContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE
        forearm = self._left_forearm(local, [-0.76, 0.25, -0.60])
        before = self._wrist_head(solver, local)
        out = local
        for _ in range(30):
            out, diag = solver.apply(local.copy(), 1 / 30)
        self.assertGreater(diag["arms"]["left"]["handBehindHead"], 0.9)
        self.assertLess(self._wrist_head(solver, out), before - 0.10)
        self.assertLess(self._wrist_head(solver, out), 0.20)
        sensed, solved = forearm[:, 0], _globals(out)[18][:, 0]
        self.assertLess(np.degrees(np.arccos(np.clip(sensed @ solved, -1, 1))), 45.0)

    def test_pelvis_rise_curls_the_upper_trunk_keeping_the_forearms(self):
        solver = CurlUpContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE
        for _ in range(30):                                   # resting levels
            out, diag = solver.apply(local.copy(), 1 / 30)
        self.assertEqual(diag["trunkCurlDeg"], 0.0)
        curl = local.copy()
        curl[0] = SUPINE @ Rotation.from_euler("x", 10, degrees=True).as_matrix()     # pelvis flexes 10 deg
        forearms = [_globals(curl)[j] for j in (18, 19)]
        out, diag = solver.apply(curl.copy(), 1 / 30)
        g = _globals(out)
        spine3 = np.degrees(np.arcsin(g[9][1, 1]))
        self.assertAlmostEqual(spine3, 18.0, delta=0.5)      # CURL_GAIN 1.8 x 10 deg above rest
        self.assertGreater(diag["trunkCurlDeg"], 7.0)
        for j, forearm in zip((18, 19), forearms):
            np.testing.assert_allclose(g[j], forearm, atol=1e-8)

    def test_forearms_rising_above_the_start_hold_curl_the_trunk(self):
        # Waist sensor flat (worn low it may even tilt down), forearms 45 deg above the start hold: full curl.
        solver = CurlUpContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE
        hold = Rotation.from_euler("z", -30, degrees=True).as_matrix()       # left forearm 30 deg below level
        mirror = np.diag([-1.0, 1.0, -1.0])
        for _ in range(20):
            solver.apply(local.copy(), 1 / 30, waist=SUPINE, forearms=(hold, mirror @ hold @ mirror))
        up = Rotation.from_euler("z", 15, degrees=True).as_matrix()          # 45 deg above the hold
        out, diag = solver.apply(local.copy(), 1 / 30, waist=SUPINE, forearms=(up, mirror @ up @ mirror))
        self.assertAlmostEqual(np.degrees(np.arcsin(_globals(out)[9][1, 1])), FOREARM_CURL_MAX_DEG, delta=1.0)

    def test_upper_back_is_kept_on_the_mat(self):
        # Lowering from a curl the network sloped the shoulder blades under the floor; a lowest-point
        # grounding then lifted the buttocks and feet.  The trunk is curled just enough to rest on the mat.
        solver = CurlUpContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE
        local[3] = Rotation.from_euler("x", -15, degrees=True).as_matrix()     # spine bent head-down
        self.assertLess(solver._lowest_back(local, 0.0), -0.02)
        out, diag = solver.apply(local.copy(), 1 / 30)
        self.assertGreater(solver._lowest_back(out, 0.0), -0.005)

    def test_forearm_toward_the_feet_is_not_pulled_to_the_head(self):
        solver = CurlUpContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE
        self._left_forearm(local, [0.0, -1.0, 0.0])
        for _ in range(10):
            out, diag = solver.apply(local.copy(), 1 / 30)
        self.assertEqual(diag["arms"]["left"]["handBehindHead"], 0.0)
        np.testing.assert_allclose(out[16], local[16], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
