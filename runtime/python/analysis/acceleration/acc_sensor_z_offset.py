"""Hypothesis: each tracker's acceleration axes are turned about the SENSOR's z axis relative to its
quaternion (an IMU/board rotation applied to one but not the other).  Lying flat on a desk (sensor z
vertical) that is only a yaw: slides still go back and forth, lifts stay vertical, as the desk tests show.
Worn on a forearm it mixes vertical and horizontal.  Scan theta: a_world' = R_raw Rz(theta) R_raw^T a_world
and score the arm raises/lowers of unity_20261103_002526 (expect about +0.4 / -0.4 m vertical).

python -m analysis.acceleration.acc_sensor_z_offset
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from production.input_recording import resolve_recording_path
from analysis.acceleration.acc_gravity_lag import integrate
from analysis.acceleration.acc_rotation_timing import ARM_MOVES


def load_raw(name, label):
    rows = [json.loads(l) for l in resolve_recording_path(name).read_text(encoding="utf-8").splitlines() if l.strip()]
    ticks = [r for r in rows if r.get("type") == "tick"]
    t = np.array([r["t"] for r in ticks]); t -= t[0]
    s = [r["samples"][label] for r in ticks]
    return (t, np.array([x["reference_adjusted_accel_xyz"] for x in s], float),
            np.array([x["accel_xyz"] for x in s], float), Rotation.from_quat([x["raw_quat_xyzw"] for x in s]),
            s[0]["tracker_key"])


def main():
    for label in ("lw", "rw"):
        t, acc_ref, acc_raw_world, R, key = load_raw("unity_20261103_002526", label)
        # the server rotates the sensor-frame acceleration with the raw rotation: accel_xyz = R_raw a_sensor
        a_sensor = R.inv().apply(acc_raw_world)
        # reference adjustment of this recording = fixed rotation from accel_xyz to reference_adjusted
        M, _ = Rotation.align_vectors(acc_ref, acc_raw_world)
        best = []
        for theta in range(0, 360, 5):
            a_w = M.apply(R.apply(Rotation.from_euler("z", theta, degrees=True).apply(a_sensor)))
            ds = []
            for a_s, b_s, sign in ARM_MOVES:
                a, b = np.searchsorted(t, a_s), np.searchsorted(t, b_s)
                ds.append(integrate(t, a_w, a, b)[-1, 1])
            err = np.sqrt(np.mean([(d - 0.4 * sg) ** 2 for d, (_, _, sg) in zip(ds, ARM_MOVES)]))
            best.append((err, theta, ds))
        best.sort(key=lambda r: r[0])
        print(f"{label} ({key}): reference adjustment {np.degrees(M.magnitude()):.0f} deg; best thetas (rms error m):")
        for err, theta, ds in best[:4]:
            print(f"   theta {theta:3d}: rms {err:.2f} | " + " ".join(f"{d:+.2f}" for d in ds))
        e0 = next(r for r in best if r[1] == 0)
        print(f"   theta   0: rms {e0[0]:.2f} | " + " ".join(f"{d:+.2f}" for d in e0[2]))


if __name__ == "__main__":
    main()
