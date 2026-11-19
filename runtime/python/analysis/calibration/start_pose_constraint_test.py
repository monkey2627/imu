import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from production.start_pose_constraint import StartPoseConstraint, _globals
from production.pose_calibration import PoseCalibrator


def knee_flex(local, hip, knee):
    global_rot = _globals(local)
    axis = np.array([0.0, -1.0, 0.0])
    thigh, shank = global_rot[hip] @ axis, global_rot[knee] @ axis
    return np.degrees(np.arccos(np.clip(thigh @ shank, -1.0, 1.0)))


def elbow_flex(local, shoulder, elbow, axis):
    global_rot = _globals(local)
    upper, forearm = global_rot[shoulder] @ axis, global_rot[elbow] @ axis
    return np.degrees(np.arccos(np.clip(upper @ forearm, -1.0, 1.0)))


class StartPoseConstraintTest(unittest.TestCase):
    labels = ["lw", "rw", "ls", "rs", "waist"]

    def tabletop_input(self):
        measured = np.tile(np.eye(3), (5, 1, 1))
        measured[2] = measured[3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        return measured

    def test_tabletop_forces_both_ambiguous_knees_to_ninety_degrees(self):
        constraint = StartPoseConstraint(self.labels)
        constraint.activate("deadbug", "tabletop")
        output, diagnostics = constraint.apply(np.tile(np.eye(3), (24, 1, 1)), self.tabletop_input())
        self.assertAlmostEqual(knee_flex(output, 1, 4), 90.0, places=5)
        self.assertAlmostEqual(knee_flex(output, 2, 5), 90.0, places=5)
        self.assertEqual(diagnostics["weights"], [1.0, 1.0])

    def test_tabletop_straightens_upper_arms_to_sensed_forearms(self):
        constraint = StartPoseConstraint(self.labels)
        constraint.activate("deadbug", "tabletop")
        pose = np.tile(np.eye(3), (24, 1, 1))
        pose[16] = pose[17] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        pose[18] = pose[19] = Rotation.from_euler("z", -90, degrees=True).as_matrix()
        output, diagnostics = constraint.apply(pose, self.tabletop_input())
        self.assertAlmostEqual(elbow_flex(output, 16, 18, np.array([1.0, 0.0, 0.0])), 0.0, places=5)
        self.assertAlmostEqual(elbow_flex(output, 17, 19, np.array([-1.0, 0.0, 0.0])), 0.0, places=5)
        self.assertEqual(diagnostics["armWeights"], [1.0, 1.0])

    def test_leg_releases_smoothly_when_its_shank_moves(self):
        constraint = StartPoseConstraint(self.labels)
        constraint.activate("deadbug", "tabletop")
        measured = self.tabletop_input()
        constraint.apply(np.tile(np.eye(3), (24, 1, 1)), measured)
        moved = measured.copy()
        moved[2] = Rotation.from_euler("x", 115, degrees=True).as_matrix()
        _, diagnostics = constraint.apply(np.tile(np.eye(3), (24, 1, 1)), moved, dt_seconds=0.1)
        self.assertAlmostEqual(diagnostics["weights"][0], 0.8)
        self.assertEqual(diagnostics["weights"][1], 1.0)

    def test_unsupported_pose_is_noop(self):
        constraint = StartPoseConstraint(self.labels)
        constraint.activate("deadbug", "supine")
        pose = np.tile(np.eye(3), (24, 1, 1))
        output, diagnostics = constraint.apply(pose, self.tabletop_input())
        np.testing.assert_array_equal(output, pose)
        self.assertFalse(diagnostics["enabled"])

    def test_committed_calibration_refreshes_lagged_measurements(self):
        calibrator = PoseCalibrator(self.labels)
        calibrator.correct(torch.from_numpy(np.tile(np.eye(3), (5, 1, 1))))
        expected = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        calibrator.corrections[:] = expected
        calibrator.refresh_corrected_history()
        np.testing.assert_allclose(calibrator.measured_frame(0),
                                   np.tile(expected, (5, 1, 1)), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
