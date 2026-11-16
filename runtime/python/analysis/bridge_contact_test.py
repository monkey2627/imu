"""python -m analysis.bridge_contact_test : lie-flat fit, trunk bias and contact solve on synthetic poses."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from production.bridge_contact import (BridgeContact, apply_trunk_bias, trunk_bias, waist_flat_fit)

SUPINE = Rotation.from_euler("x", -90, degrees=True).as_matrix()     # chest normal (z) up, head (y) horizontal


class BridgeTest(unittest.TestCase):
    def test_waist_flat_fit_keeps_heading(self):
        tilted = Rotation.from_euler("y", 40, degrees=True).as_matrix() @ SUPINE @ Rotation.from_euler(
            "x", 15, degrees=True).as_matrix()
        c, tilt = waist_flat_fit(tilted)
        flat = tilted @ c
        self.assertAlmostEqual(flat[1, 2], 1.0, places=6)          # chest normal up
        self.assertAlmostEqual(flat[1, 1], 0.0, places=6)          # spine axis horizontal
        head = tilted[:, 1].copy(); head[1] = 0; head /= np.linalg.norm(head)
        self.assertGreater(flat[:, 1] @ head, 0.999)               # same heading
        self.assertAlmostEqual(tilt, 15.0, delta=0.5)

    def test_trunk_bias_removes_constant_offset_and_keeps_spine1(self):
        rng = np.random.default_rng(0)
        local = np.tile(np.eye(3), (40, 24, 1, 1))
        local[:, 0] = SUPINE
        bias = {j: Rotation.from_euler("x", 8 + j, degrees=True).as_matrix() for j in (3, 6, 9)}
        for j, r in bias.items():
            local[:, j] = r @ Rotation.from_rotvec(rng.normal(0, 0.01, (40, 3))).as_matrix()
        est, spread = trunk_bias(local)
        self.assertLess(spread, 3)
        out = apply_trunk_bias(local[0], est)
        g_spine1_before, g_spine1_after = local[0, 0] @ local[0, 3], out[0] @ out[3]
        self.assertTrue(np.allclose(g_spine1_before, g_spine1_after, atol=1e-9))
        for j in (3, 6, 9):
            angle = np.degrees(np.arccos(np.clip((np.trace(out[j]) - 1) / 2, -1, 1)))
            self.assertLess(angle, 2.5)

    def test_contact_planted_feet(self):
        solver = BridgeContact()
        local = np.tile(np.eye(3), (24, 1, 1))
        local[0] = SUPINE
        # thigh straight along the body, shank pointing down 60 deg: the solve must raise the thigh so the
        # ankle sits at foot-flat height.
        for hip, knee in ((1, 4), (2, 5)):
            local[hip] = Rotation.from_euler("x", 0, degrees=True).as_matrix()
            local[knee] = Rotation.from_euler("x", 150, degrees=True).as_matrix()
        for _ in range(60):
            out, diag = solver.apply(local, 2 / 60)
        self.assertEqual(diag["left"]["weight"], 1.0)
        g = [None] * 24
        from production.arm_prior import PARENTS
        for j in range(24):
            g[j] = out[j] if PARENTS[j] < 0 else g[PARENTS[j]] @ out[j]
        pos = solver._positions(np.stack(g), diag["pelvisHeightM"])
        self.assertAlmostEqual(pos[7, 1], 0.077, delta=0.01)
        self.assertTrue(np.allclose(g[4], local[0] @ local[1] @ local[4], atol=1e-9))   # shank global kept


if __name__ == "__main__":
    unittest.main()
