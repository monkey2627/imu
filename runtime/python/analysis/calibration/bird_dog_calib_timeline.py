"""Timeline of a bird-dog calibration recording: per 0.25 s, the waist forward (azimuth, pitch of
the spine), each forearm and shank direction (azimuth relative to the waist forward, elevation),
raw input, plus calibration events.  python -m analysis.calibration.bird_dog_calib_timeline <recording>
"""
import json
import sys
import numpy as np

from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand
from production.pose_calibration import BONE_AXES, _best_yaw


def main():
    path = resolve_recording_path(sys.argv[1])
    rec = load_input_recording(path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    quats, _, events = _expand(rec, rows)
    ev = {f: e["name"] for f, e in events}
    all_events = [r for r in rows if r.get("type") == "event"]
    t0 = rec.times[0]
    for e in all_events:
        print(f"  event {e['t'] - t0:6.2f}s {e['name']} {e.get('message', '')[:120]}")
    ori = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats]).astype(np.float64)
    wi = rec.labels.index("waist")
    print("t | waist yaw, spine pitch | lw az/el | rw az/el | ls az/el | rs az/el   (az rel. waist forward, + = toward body-left)")
    for t in range(0, len(ori), 15):
        yaw = _best_yaw([ori[t, wi]])
        spine = ori[t, wi] @ np.array([0.0, 1.0, 0.0])
        line = f"{t/60:6.2f}s | {np.degrees(yaw):+5.0f} {np.degrees(np.arcsin(np.clip(spine[2]*np.cos(yaw)+spine[0]*np.sin(yaw), -1, 1))):+4.0f}"
        for lab in ("lw", "rw", "ls", "rs"):
            d = ori[t, rec.labels.index(lab)] @ BONE_AXES[lab]
            az = np.degrees((np.arctan2(d[0], d[2]) - yaw + np.pi) % (2 * np.pi) - np.pi)
            line += f" | {lab} {az:+5.0f} {np.degrees(np.arcsin(np.clip(d[1], -1, 1))):+4.0f}"
        if any(t <= f < t + 15 for f in ev):
            line += "  <- " + ", ".join(n for f, n in ev.items() if t <= f < t + 15)
        print(line)


if __name__ == "__main__":
    main()
