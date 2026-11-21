"""Run with python -m analysis.calibration.exercise_calibration_synthetic.

Synthetic direction observations with independent world headings, fixed local
mount errors and noisy holds. No model weights or hardware are required.
"""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from production.exercise_calibration import EXERCISES, POSES, ExerciseCalibration
from production.pose_calibration import BONE_AXES, PoseCalibrator, _rotation_between, _yaw
from production.web_visualizer import _render_html

LABELS = ["lw", "rw", "ls", "rs", "waist"]


def observations(cal, pose, facing=37, error=True):
    waist = _yaw(np.radians(facing)) @ Rotation.from_euler(
        "z" if pose.startswith("side_") else "x",
        90 if pose.startswith(("side_", "cat_")) else -90, degrees=True).as_matrix()
    head = waist[:, 1].copy()
    head[1] = 0
    head /= np.linalg.norm(head)
    axes = {"-h": -head, "+h": head, "+c": waist[:, 2], "up": np.array([0., 1., 0.]), "down": np.array([0., -1., 0.])}
    values = []
    for i, label in enumerate(LABELS[:-1]):
        target = axes[POSES[pose].get(label, ("-h", "check"))[0]]
        true = _rotation_between(BONE_AXES[label], target)
        # Fixed per-sensor mount rotation; each pose has a different true orientation.
        mount = Rotation.from_euler("xyz", [4, -3, 5], degrees=True).as_matrix() if error else np.eye(3)
        heading = _yaw(np.radians([115, -70, 32, -28][i])) if error else np.eye(3)
        values.append(heading @ true @ mount)
    return np.stack(values + [waist])


def feed(cal, matrices, frames=60, noisy=False):
    rng = np.random.default_rng(123)
    for _ in range(frames):
        noise = Rotation.from_rotvec(rng.normal(0, .003, (5, 3))).as_matrix() if noisy else np.tile(np.eye(3), (5, 1, 1))
        cal.correct(torch.from_numpy(noise @ matrices))


class CalibrationChecks(unittest.TestCase):
    def test_each_action_selects_a_final_start_pose_hold(self):
        for action, config in EXERCISES.items():
            for start_pose, expected in config["start_poses"].items():
                with self.subTest(action=action, start_pose=start_pose):
                    sequence = ExerciseCalibration(PoseCalibrator(LABELS))
                    sequence.start(action, start_pose)
                    self.assertEqual(sequence.start_pose, start_pose)
                    self.assertEqual(sequence.steps[-1]["pose"], expected["pose"])
                    self.assertEqual(sequence.status()["startPose"], start_pose)
        sequence = ExerciseCalibration(PoseCalibrator(LABELS))
        with self.assertRaises(ValueError):
            sequence.start("deadbug", "not-a-pose")

    def test_lyingcat_is_one_fixed_end_neutral_hold(self):
        steps = EXERCISES["lyingcat"]["steps"]
        self.assertEqual([(item["pose"], item["seconds"]) for item in steps], [("cat_neutral", 4)])
        self.assertEqual(set(POSES["cat_neutral"]), set(BONE_AXES))
        self.assertEqual(POSES["cat_neutral"]["lw"], ("down", "fit"))
        self.assertEqual(POSES["cat_neutral"]["rw"], ("down", "fit"))

    def test_all_actions_and_facings(self):
        for action, config in EXERCISES.items():
            for facing in (37, -140):
                with self.subTest(action=action, facing=facing):
                    cal = PoseCalibrator(LABELS)
                    sequence = ExerciseCalibration(cal)
                    sequence.start(action)
                    fitted = set()
                    for current in config["steps"]:
                        cal._history.clear()
                        feed(cal, observations(cal, current["pose"], facing), noisy=True)
                        passed, message = sequence.capture(current["pose"], current["seconds"])
                        self.assertTrue(passed, message)
                        fitted |= {label for label, (_, op) in POSES[current["pose"]].items() if op == "fit"}
                    self.assertEqual(fitted, set(BONE_AXES))

    def test_cancel_switch_and_failed_step_restore(self):
        cal = PoseCalibrator(LABELS)
        cal.headings[:] = _yaw(.2)
        before = cal.headings.copy(), cal.corrections.copy()
        sequence = ExerciseCalibration(cal)
        sequence.start("hipthrust")
        feed(cal, observations(cal, "supine_straight"))
        sequence.elapsed = 13 * 60
        sequence.advance(60)
        self.assertTrue(sequence.active)
        sequence.start("deadbug")  # cancels and restores before starting the other action
        sequence.cancel()
        np.testing.assert_allclose(cal.headings, before[0])
        np.testing.assert_allclose(cal.corrections, before[1])
        sequence.start("crunch")
        # Upright trunk must never be accepted as a lying hold.
        values = observations(cal, "supine_straight")
        values[-1] = np.eye(3)
        feed(cal, values)
        sequence.elapsed = 13 * 60
        sequence.advance(60)
        self.assertEqual(sequence.outcome, "failed")
        np.testing.assert_allclose(cal.headings, before[0])
        np.testing.assert_allclose(cal.corrections, before[1])

    def test_simple_start_aligns_level_forearm_heading_only(self):
        # Live 2026-11-19: the right forearm read 177° toward the head while lying supine.
        # Default start poses use full templates (start_pose_fit_test.py); start poses without a template
        # (here the raised hip thrust) still align only the level forearm heading.
        for action, start_pose, pose in (("hipthrust", "raised", "bridge_check"),):
            with self.subTest(action=action):
                cal = PoseCalibrator(LABELS)
                sequence = ExerciseCalibration(cal, simple_start_only=True)
                sequence.start(action, start_pose)
                feed(cal, observations(cal, pose, error=False))
                values = observations(cal, pose, error=False)
                values[1] = _yaw(np.pi) @ values[1]
                feed(cal, values, 120)
                passed, message = sequence.capture(pose, 2)
                self.assertTrue(passed, message)
                self.assertAlmostEqual(abs(np.degrees(np.arctan2(cal.headings[1][0, 2], cal.headings[1][0, 0]))),
                                       180.0, delta=1.0)
                np.testing.assert_allclose(cal.corrections, np.tile(np.eye(3), (5, 1, 1)))
                np.testing.assert_allclose(cal.headings[2:], np.tile(np.eye(3), (3, 1, 1)))
        # A forearm pointing up cannot be fixed by a yaw: still a failed check.
        cal = PoseCalibrator(LABELS)
        sequence = ExerciseCalibration(cal, simple_start_only=True)
        sequence.start("hipthrust", "raised")
        values = observations(cal, "bridge_check", error=False)
        values[1] = _rotation_between(BONE_AXES["rw"], np.array([0., 1., 0.]))
        feed(cal, values, 120)
        passed, message = sequence.capture("bridge_check", 2)
        self.assertFalse(passed, message)
        np.testing.assert_allclose(cal.headings, np.tile(np.eye(3), (5, 1, 1)))

    def test_moving_missing_and_disabled(self):
        cal = PoseCalibrator(LABELS)
        sequence = ExerciseCalibration(cal)
        sequence.start("hipthrust")
        pose = observations(cal, "supine_straight")
        for i in range(60):
            values = pose.copy()
            values[0] = _yaw(np.radians(20 if i % 2 else -20)) @ values[0]
            feed(cal, values, 1)
        passed, message = sequence.capture("supine_straight", 10)
        self.assertFalse(passed, message)
        self.assertIn("lw", message)
        for labels in (LABELS[:-1], ["lw", "rw", "waist"]):
            with self.assertRaises(ValueError):
                ExerciseCalibration(PoseCalibrator(labels)).start("deadbug")
        cal.calibration_enabled = False
        with self.assertRaises(ValueError):
            sequence.start("deadbug")

    def test_timer_completion(self):
        cal = PoseCalibrator(LABELS)
        sequence = ExerciseCalibration(cal)
        sequence.start("hipthrust")
        for current in EXERCISES["hipthrust"]["steps"]:
            feed(cal, observations(cal, current["pose"]))
            sequence.elapsed = (current["seconds"] + 3) * 60
            event = sequence.advance(60)
            self.assertEqual(event["pose"], current["pose"])
        self.assertEqual(sequence.outcome, "complete")
        self.assertFalse(sequence.active)

    def test_failed_cross_check_restores_entire_previous_set(self):
        cal = PoseCalibrator(LABELS)
        cal.headings[:] = _yaw(.15)
        previous = cal.headings.copy(), cal.corrections.copy()
        sequence = ExerciseCalibration(cal)
        sequence.start("hipthrust")
        feed(cal, observations(cal, "supine_straight"))
        sequence.elapsed = 13 * 60
        sequence.advance(60)
        wrong = observations(cal, "supine_check")
        wrong[0] = _yaw(np.radians(60)) @ wrong[0]
        feed(cal, wrong, 8 * 60)
        sequence.elapsed = 11 * 60
        sequence.advance(60)
        self.assertEqual(sequence.outcome, "failed")
        self.assertIn("CHECK FAILED", sequence.results[-1])
        np.testing.assert_allclose(cal.headings, previous[0])
        np.testing.assert_allclose(cal.corrections, previous[1])

    def test_cat_reach_uses_neutral_heading(self):
        cal = PoseCalibrator(LABELS)
        sequence = ExerciseCalibration(cal)
        sequence.start("lyingcat")
        feed(cal, observations(cal, "cat_shanks", error=False))
        self.assertTrue(sequence.capture("cat_shanks", 8)[0])
        cal._history.clear()
        reached = observations(cal, "cat_left", error=False)
        reached[-1] = _yaw(np.radians(20)) @ reached[-1]
        feed(cal, reached)
        self.assertTrue(sequence.capture("cat_left", 8)[0])
        np.testing.assert_allclose(cal.headings[0], np.eye(3), atol=1e-7)

    def test_cat_check_allows_small_wrist_reseat_error_but_rolls_back_large_error(self):
        cal = PoseCalibrator(LABELS)
        sequence = ExerciseCalibration(cal)
        sequence.start("lyingcat")
        for pose, seconds in (("cat_shanks", 10), ("cat_left", 8), ("cat_right", 8)):
            feed(cal, observations(cal, pose, error=False), seconds * 60)
            passed, message = sequence.capture(pose, seconds)
            self.assertTrue(passed, message)
            cal._history.clear()

        check = observations(cal, "cat_check", error=False)
        target = Rotation.from_euler("x", 32, degrees=True).as_matrix()
        check[0] = target @ check[0]
        check[1] = target @ check[1]
        feed(cal, check, 10 * 60)
        passed, message = sequence.capture("cat_check", 10)
        self.assertTrue(passed, message)
        self.assertIn("复核32°", message)

        previous = cal.headings.copy(), cal.corrections.copy()
        check[0] = Rotation.from_euler("x", 42, degrees=True).as_matrix() @ observations(cal, "cat_check", error=False)[0]
        check[1] = Rotation.from_euler("x", 42, degrees=True).as_matrix() @ observations(cal, "cat_check", error=False)[1]
        cal._history.clear()
        feed(cal, check, 10 * 60)
        passed, message = sequence.capture("cat_check", 10)
        self.assertFalse(passed, message)
        self.assertIn("CHECK FAILED", message)
        np.testing.assert_allclose(cal.headings, previous[0])
        np.testing.assert_allclose(cal.corrections, previous[1])

    def test_html_catalog(self):
        page = _render_html(21200, [])
        self.assertNotIn("__EXERCISE_CATALOG__", page)
        self.assertIn('id="exerciseSelect"', page)
        for action in EXERCISES:
            self.assertIn('"' + action + '"', page)

    def test_offline_replays_capture_and_rollback_events(self):
        from production.offline_compare import calibrated_orientations
        from production.calibration import _bone_offset, SERVER_TO_SMPL
        basis = SERVER_TO_SMPL.numpy()
        for outcome in ("complete", "failed"):
            cal = PoseCalibrator(LABELS)
            sequence = ExerciseCalibration(cal)
            initial = np.tile(_yaw(.12), (5, 1, 1))
            cal.headings[:] = initial
            sequence.start("hipthrust")
            events = [(0, {"name": "exerciseCalibrationStart", "action": "hipthrust",
                           "initialHeadings": initial.tolist(), "initialCorrections": cal.corrections.tolist()})]
            raw = [observations(cal, "supine_straight")]
            for current in EXERCISES["hipthrust"]["steps"]:
                matrices = observations(cal, current["pose"])
                raw.extend([matrices] * (current["seconds"] * 60))
                feed(cal, matrices, current["seconds"] * 60)
                sequence.capture(current["pose"], current["seconds"])
                events.append((len(raw) - 1, {"name": "exercisePoseCalibrate", **current}))
            events.append((len(raw) - 1, {"name": "exerciseCalibrationEnd", "outcome": outcome}))
            if outcome == "failed":
                sequence.cancel()
            quats = np.asarray([Rotation.from_matrix(np.stack([
                basis @ matrix @ basis @ _bone_offset(label).numpy().T
                for matrix, label in zip(frame, LABELS)])).as_quat() for frame in raw])
            _, replayed, messages = calibrated_orientations(SimpleNamespace(labels=LABELS), quats, events)
            self.assertEqual(len(messages), 4)
            np.testing.assert_allclose(replayed.headings, cal.headings, atol=1e-6)
            np.testing.assert_allclose(replayed.corrections, cal.corrections, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
