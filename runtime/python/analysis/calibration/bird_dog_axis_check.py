"""Independent heading check: the base->reach arm raise is a rotation about the body's lateral axis.
A world heading error H rotates the measured rotation axis by H, so the azimuth offset of the
measured axis vs. the waist lateral axis estimates H without assuming where the hand ended up.
python -m analysis.calibration.bird_dog_axis_check <recording> ...
"""
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand
from production.pose_calibration import BONE_AXES, _best_yaw

for name in sys.argv[1:]:
    path = resolve_recording_path(name)
    rec = load_input_recording(path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    quats, _, events = _expand(rec, rows)
    ev = {e["name"]: f for f, e in events}
    ori = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats]).astype(np.float64)
    mean = lambda lab, f: Rotation.from_matrix(ori[f - 30:f, rec.labels.index(lab)]).mean().as_matrix()
    print(f"== {path.name}")
    for lab, reach in (("lw", "birdDogLeftReachCalibrate"), ("rw", "birdDogRightReachCalibrate")):
        b, r = ev["birdDogBaseCalibrate"], ev[reach]
        base, end = mean(lab, b), mean(lab, r)
        yaw = _best_yaw([mean("waist", r)])
        rv = Rotation.from_matrix(end @ base.T).as_rotvec()          # world-frame delta
        ang = np.degrees(np.linalg.norm(rv)); axis = rv / np.linalg.norm(rv)
        horiz = np.degrees(np.arcsin(abs(axis[1])))
        # true axis for raising the arm forward from hanging: about body -x for... take either sign
        lateral = np.arctan2(1.0, 0.0) + yaw                         # azimuth of body +x
        off = np.degrees((np.arctan2(axis[0], axis[2]) - lateral + np.pi) % (2 * np.pi) - np.pi)
        off = (off + 90) % 180 - 90                                   # axis sign ambiguity -> (-90, 90]
        d = end @ BONE_AXES[lab]
        az = np.degrees((np.arctan2(d[0], d[2]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        print(f"  {lab}: raise {ang:4.0f} deg, axis tilt from horizontal {horiz:3.0f}, axis azimuth offset {off:+5.0f}"
              f" | reach bone az {az:+5.0f} (=> heading fix {-az:+5.0f})")
