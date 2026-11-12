"""Replay a recording's calibration events through the current PoseCalibrator and print its messages.
python -m analysis.calibration.replay_calibration <recording> ...
"""
import json, sys
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand, calibrated_orientations

for name in sys.argv[1:]:
    path = resolve_recording_path(name)
    rec = load_input_recording(path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    quats, _, events = _expand(rec, rows)
    _, cal, msgs = calibrated_orientations(rec, quats, events)
    print(f"== {path.name}")
    for t, m in msgs:
        print(f"  {t:5.1f}s {m[:200]}")
