import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from production.start_pose_constraint import StartPoseConstraint, _globals
from production.start_pose_rebase import StartPoseRebaser


class StartPoseRebaseTest(unittest.TestCase):
    labels = ["lw", "rw", "ls", "rs", "waist"]

    def measured(self):
        value = np.tile(np.eye(3), (5, 1, 1))
        value[2] = value[3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        return value

    def test_first_frame_is_exact_tabletop_template(self):
        pose = np.tile(np.eye(3), (24, 1, 1))
        pose[16] = pose[17] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        pose[18] = pose[19] = Rotation.from_euler("z", -90, degrees=True).as_matrix()
        measured = self.measured()

        expected_builder = StartPoseConstraint(self.labels)
        expected_builder.activate("deadbug", "tabletop")
        expected, _ = expected_builder.apply(pose, measured, dt_seconds=0.0)

        rebaser = StartPoseRebaser(self.labels)
        rebaser.activate("deadbug", "tabletop")
        actual, diag = rebaser.apply(pose, measured)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertTrue(diag["captured"])

    def test_each_frame_uses_fixed_origin_without_accumulation(self):
        pose = np.tile(np.eye(3), (24, 1, 1))
        measured = self.measured()
        rebaser = StartPoseRebaser(self.labels)
        rebaser.activate("deadbug", "tabletop")
        rebaser.apply(pose, measured)

        moved_pose = pose.copy()
        moved_pose[1] = Rotation.from_euler("z", 10, degrees=True).as_matrix()
        moved_measured = measured.copy()
        moved_measured[2] = Rotation.from_euler("x", 120, degrees=True).as_matrix()
        first, _ = rebaser.apply(moved_pose, moved_measured)
        second, _ = rebaser.apply(moved_pose, moved_measured)
        np.testing.assert_allclose(first, second, atol=1e-7)

        global_rot = _globals(first)
        np.testing.assert_allclose(global_rot[4], moved_measured[2], atol=1e-7)
        expected_hip = rebaser.target_global[1] @ Rotation.from_euler(
            "z", 10, degrees=True).as_matrix()
        np.testing.assert_allclose(global_rot[1], expected_hip, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
