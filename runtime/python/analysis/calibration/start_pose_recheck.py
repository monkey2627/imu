"""Re-run the start-pose check of recorded exercise calibrations with the current code.

    python -m analysis.calibration.start_pose_recheck calibration_20261119_210011.jsonl ...
"""
import json, sys, numpy as np, torch
from production.deadbug_replay import replay
from production.exercise_calibration import ExerciseCalibration
from production.pose_calibration import PoseCalibrator
from production.paths import paths
torch.set_grad_enabled(False)
for name in sys.argv[1:]:
    path = paths.recordings/name
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    starts = [r for r in rows if r.get("name") == "exerciseCalibrationStart"]
    cap = next(r for r in rows if r.get("name") == "exercisePoseCalibrate")
    r = replay(path, model_id="waist_imudino_stream_curlup", ignore_recorded_calibration=True)
    O = r.calibrated_ori
    cal = PoseCalibrator(r.labels)
    seq = ExerciseCalibration(cal, simple_start_only=True)
    seq.start(starts[-1]["action"], starts[-1]["startPose"])
    end = min(len(O), int(round(cap["t"] * 60)))
    cal._history.extend(O[:end])
    ok, msg = seq.capture(cap["pose"], cap["seconds"])
    print(name, starts[-1]["action"], "PASS" if ok else "FAIL", "|", msg)
