"""python -m analysis.curlup_contact_test"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from production.arm_prior import PARENTS, _globals
from production.bridge_contact import PELVIS_BACK_DEPTH
from production.curlup_contact import ARM_MAX_SPEED_DEG_S, CurlUpContact

SUPINE = Rotation.from_euler("x", -90, degrees=True).as_matrix()


class CurlUpContactTest(unittest.TestCase):
    def test_pelvis_contact_does_not_force_shoulders_down(self):
        solver = CurlUpContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE
        local[3] = Rotation.from_euler("x", 25, degrees=True).as_matrix()
        out, diag = solver.apply(local, 1 / 30)
        self.assertAlmostEqual(diag["pelvisHeightM"], PELVIS_BACK_DEPTH, delta=0.002)
        self.assertTrue(np.allclose(out[3], local[3]))

    def test_upper_arm_jump_is_limited_and_forearm_is_preserved(self):
        solver = CurlUpContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE
        solver.apply(local, 1 / 30)
        jumped = local.copy()
        jumped[16] = Rotation.from_euler("z", 100, degrees=True).as_matrix()
        before = _globals(jumped)[18]
        out, diag = solver.apply(jumped, 1 / 30)
        after = _globals(out)[18]
        self.assertLessEqual(diag["arms"]["left"]["appliedDeltaDeg"], ARM_MAX_SPEED_DEG_S / 30 + 0.1)
        self.assertTrue(np.allclose(before, after, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
