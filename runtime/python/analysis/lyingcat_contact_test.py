"""python -m analysis.lyingcat_contact_test"""
import math
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from production.arm_prior import PARENTS, _globals, _rotation_between
from production.lyingcat_contact import LyingCatContact

DOWN = np.array([0.0, -1.0, 0.0])
# Prone body: SMPL +y (head) -> +z, front (+z) -> -y (facing the mat).
ROOT = Rotation.from_euler("x", 90, degrees=True).as_matrix()


def _rx(deg):
    return Rotation.from_euler("x", deg, degrees=True).as_matrix()


def _locals(globals_):
    return np.asarray([globals_[j] if p < 0 else globals_[p].T @ globals_[j] for j, p in enumerate(PARENTS)])


def _quadruped(solver, thigh_tilt=(25.0, -20.0), upper_tilt=(35.0, -15.0), lifted_right_arm=False):
    """All-fours pose with forearms down and shanks level, but slanted unobserved thighs / upper arms."""
    o = solver.offsets
    g = [ROOT.copy() for _ in range(24)]
    for (upper, forearm, wrist, hand), tilt in zip(((16, 18, 20, 22), (17, 19, 21, 23)), upper_tilt):
        down = _rotation_between(o[forearm], DOWN)
        g[upper] = _rx(tilt) @ down
        g[forearm] = g[wrist] = g[hand] = down
    if lifted_right_arm:          # right forearm reaching forward (head-ward, horizontal)
        g[19] = g[21] = g[23] = _rotation_between(o[21], np.array([0.0, 0.0, 1.0]))
    for (thigh, shank, ankle, foot), tilt in zip(((1, 4, 7, 10), (2, 5, 8, 11)), thigh_tilt):
        g[thigh] = _rx(tilt) @ _rotation_between(o[shank], DOWN)
        g[shank] = g[ankle] = g[foot] = ROOT      # shank -y -> -z: level, pointing to the feet
    return _locals(g)


def _elevation(v):
    v = v / np.linalg.norm(v)
    return math.degrees(math.asin(float(np.clip(v[1], -1, 1))))


class LyingCatContactTest(unittest.TestCase):
    def test_thighs_and_upper_arms_vertical_supports_share_plane(self):
        solver = LyingCatContact()
        local = _quadruped(solver)
        before = _globals(local)
        for _ in range(60):
            out, diag = solver.apply(local, 1 / 30)
        after = _globals(out)
        p = solver._positions(after)
        o = solver.offsets
        for key, value in diag["weights"].items():
            self.assertEqual(value, 1.0, key)
        for thigh, knee in ((1, 4), (2, 5)):
            self.assertLess(_elevation(after[thigh] @ o[knee]), -80.0)
        for upper, elbow in ((16, 18), (17, 19)):
            self.assertLess(_elevation(after[upper] @ o[elbow]), -60.0)
        floor = diag["supportPlaneM"]
        for joint in (20, 21, 4, 5):
            self.assertLess(abs(p[joint, 1] - floor), 0.015, joint)
        # Sensed forearms keep their measurement exactly.  Shanks are only levelled: the SMPL
        # knee->ankle offset is ~5 deg off the shank sensor's -y axis, the same tilt seen live.
        for joint in (18, 19):
            np.testing.assert_allclose(after[joint], before[joint], atol=1e-6)
        for shank, ankle in ((4, 7), (5, 8)):
            self.assertLess(abs(p[shank, 1] - p[ankle, 1]), 1e-4)
            angle = math.degrees(math.acos(np.clip((np.trace(before[shank].T @ after[shank]) - 1) / 2, -1, 1)))
            self.assertLess(angle, 8.0)
        # Palms and foot dorsums lie flat.
        for a, b in ((20, 22), (21, 23), (7, 10), (8, 11)):
            self.assertLess(abs(p[a, 1] - p[b, 1]), 0.01)

    def test_lifted_arm_is_released_and_keeps_its_measurement(self):
        solver = LyingCatContact()
        local = _quadruped(solver, lifted_right_arm=True)
        before = _globals(local)
        for _ in range(60):
            out, diag = solver.apply(local, 1 / 30)
        after = _globals(out)
        self.assertEqual(diag["weights"]["rightArm"], 0.0)
        self.assertEqual(diag["weights"]["leftArm"], 1.0)
        np.testing.assert_allclose(after[19], before[19], atol=1e-6)
        self.assertNotIn("rightWrist", diag["afterErrorCm"])
        self.assertLess(abs(diag["afterErrorCm"]["leftWrist"]), 1.5)

    def test_measured_forearm_overrides_network(self):
        solver = LyingCatContact()
        local = _quadruped(solver)
        measured_left = _rotation_between(solver.offsets[20], np.array([0.2, -1.0, 0.1]))
        network = local.copy()
        network[18] = np.eye(3)            # network forearm wrong
        labels = ["lw"]
        out, _ = solver.apply(network, 1 / 30, measured=np.asarray([measured_left]), labels=labels)
        np.testing.assert_allclose(_globals(out)[18], measured_left, atol=1e-6)

    def test_weights_are_rate_limited(self):
        solver = LyingCatContact()
        _, diag = solver.apply(_quadruped(solver), 1 / 30)
        for value in diag["weights"].values():
            self.assertLessEqual(value, 4.0 / 30 + 1e-9)

    def test_runtime_budget(self):
        solver = LyingCatContact()
        local = _quadruped(solver)
        solver.apply(local, 1 / 30)
        start = time.perf_counter()
        for _ in range(30):
            solver.apply(local, 1 / 30)
        self.assertLess((time.perf_counter() - start) / 30, 0.015)


if __name__ == "__main__":
    unittest.main()
