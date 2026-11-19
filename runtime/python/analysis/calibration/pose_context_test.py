"""Run with: python -m analysis.calibration.pose_context_test"""
import unittest

import numpy as np
import torch

from production.pose_calibration import PoseCalibrator, _yaw
from production.pose_context import corrected_orientation, replay_start_context


LABELS = ["lw", "rw", "ls", "rs", "waist"]


class FeatureSession:
    def __init__(self):
        self.features = []

    def push_feature(self, feature):
        self.features.append(np.asarray(feature).copy())


class PoseContextChecks(unittest.TestCase):
    def test_replay_uses_final_calibration_and_requested_tail(self):
        calibrator = PoseCalibrator(LABELS)
        calibrator.headings[:] = _yaw(np.radians(20))
        raw = np.tile(np.eye(3, dtype=np.float32), (5, 1, 1))
        expected = corrected_orientation(raw, calibrator).numpy()
        observations = []
        for frame in range(5):
            feature = np.zeros(60, dtype=np.float32)
            feature[0] = frame
            observations.append((feature, raw, np.zeros((5, 3), dtype=np.float32)))
        session = FeatureSession()
        count = replay_start_context(
            session, observations, "waist_mobileposer_amass", LABELS, calibrator, frames=3)
        self.assertEqual(count, 3)
        self.assertEqual(len(session.features), 3)
        self.assertEqual([row[0] for row in session.features], [2, 3, 4])
        np.testing.assert_allclose(session.features[-1][-45:].reshape(5, 3, 3), expected, atol=1e-6)

    def test_empty_context_is_rejected(self):
        with self.assertRaises(ValueError):
            replay_start_context(
                FeatureSession(), [], "waist_mobileposer_amass", LABELS, PoseCalibrator(LABELS))


if __name__ == "__main__":
    unittest.main()
