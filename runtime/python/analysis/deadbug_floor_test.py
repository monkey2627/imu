"""python -m analysis.deadbug_floor_test"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from production.arm_prior import _globals
from production.deadbug_floor import DeadBugFloorContact

SUPINE = Rotation.from_euler("x", -90, degrees=True).as_matrix()   # SMPL front (+z) -> up (+y)


def _rx(deg):
    return Rotation.from_euler("x", deg, degrees=True).as_matrix()


def _lowest(solver, local, region):
    g = np.asarray(_globals(local))
    return solver._points(region, g, solver._positions(g))[:, 1].min()


class DeadBugFloorTest(unittest.TestCase):
    def setUp(self):
        self.solver = DeadBugFloorContact()
        self.flat = np.tile(np.eye(3), (24, 1, 1))
        self.flat[0] = SUPINE

    def _run(self, local, frames=60):
        self.solver.reset("test")
        for _ in range(frames):
            out, diag = self.solver.apply(local, 1 / 30)
        return out, diag

    def test_flat_supine_is_untouched(self):
        out, diag = self._run(self.flat)
        self.assertEqual(diag["weight"], 1.0)
        np.testing.assert_allclose(out, self.flat, atol=1e-9)

    def test_leg_below_the_back_is_lifted_and_shank_kept(self):
        for tilt in (30.0, -30.0):
            local = self.flat.copy()
            local[2] = _rx(tilt)
            floor = _lowest(self.solver, local, "back")
            if _lowest(self.solver, local, "rightLeg") >= floor - 0.01:
                continue                       # this sign raises the leg: nothing to fix
            before = np.asarray(_globals(local))
            out, diag = self._run(local)
            after = np.asarray(_globals(out))
            self.assertGreater(abs(diag["rightLeg"]["appliedDeg"]), 5.0)
            self.assertGreaterEqual(_lowest(self.solver, out, "rightLeg"), floor - 0.01)
            np.testing.assert_allclose(after[5], before[5], atol=1e-9)    # measured shank unchanged
            self.assertAlmostEqual(diag["leftLeg"]["appliedDeg"], 0.0, places=6)
            return
        self.fail("neither hip direction put the leg below the back")

    def test_head_tipped_back_is_tucked(self):
        for tilt in (40.0, -40.0):
            local = self.flat.copy()
            local[12] = _rx(tilt)
            floor = _lowest(self.solver, local, "back")
            if _lowest(self.solver, local, "head") >= floor - 0.01:
                continue
            out, diag = self._run(local)
            self.assertGreaterEqual(_lowest(self.solver, out, "head"), floor - 0.01)
            return
        self.fail("neither neck direction put the head below the back")

    def test_not_supine_does_nothing(self):
        standing = np.tile(np.eye(3), (24, 1, 1))
        standing[2] = _rx(-60)
        out, diag = self._run(standing)
        self.assertEqual(diag["weight"], 0.0)
        np.testing.assert_allclose(out, standing, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
