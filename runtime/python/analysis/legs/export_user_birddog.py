"""Export a bird-dog calibration recording's calibrated input (SMPL frame) for server-side model tests.
python -m analysis.legs.export_user_birddog <recording> <out.npz>
"""
import json, sys
import numpy as np
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand, calibrated_orientations

path = resolve_recording_path(sys.argv[1])
rec = load_input_recording(path)
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
quats, accels, events = _expand(rec, rows)
raw = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats]).astype(np.float64)
_, cal, _ = calibrated_orientations(rec, quats, events)
ori = (cal.headings[None] @ raw @ cal.corrections[None]).astype(np.float32)
acc = (accels @ np.diag([-1.0, 1.0, -1.0]).T).astype(np.float32)        # server -> SMPL world frame
ev = {e["name"]: f for f, e in events}
np.savez(sys.argv[2], ori=ori, acc=acc, labels=np.array(rec.labels),
         left_reach=ev["birdDogLeftReachCalibrate"], right_reach=ev["birdDogRightReachCalibrate"],
         base=ev["birdDogBaseCalibrate"])
print(ori.shape, acc.shape, {k: v for k, v in ev.items()})
