"""Hypothesis: the world-frame acceleration was rotated with an orientation from a different moment than
the one it was measured at.  Harmless when the tracker only translates (desk slides) or only rotates with
~zero acceleration (desk tilts); wrong when it rotates AND accelerates (an arm raise, a shank lift).
Re-rotate a_world(t) by R(t + tau) R(t)^T and look for the tau that makes the arm raises/lowers of
unity_20261103_002526 integrate to the expected vertical displacement (~+0.4 / -0.4 m), checking that the
desk test (unity_20261106_200542 slides) stays correct.

python -m analysis.acceleration.acc_rotation_timing
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from analysis.acceleration.acc_gravity_lag import load, integrate, segments

ARM_MOVES = [(2.6, 4.1, +1), (5.2, 7.1, -1), (8.6, 10.0, +1), (11.1, 12.3, -1), (13.6, 14.8, +1),
             (15.9, 17.2, -1), (17.9, 19.1, +1), (20.3, 22.0, -1)]


def rerotate(t, acc, q, tau):
    R = Rotation.from_quat(q)
    shifted = Slerp(t, R)(np.clip(t + tau, t[0], t[-1]))
    return (shifted * R.inv()).apply(acc)


def arm_score(tau):
    errs, rows = [], {}
    for label in ("lw", "rw"):
        t, acc, q = load("unity_20261103_002526", label)
        a2 = rerotate(t, acc, q, tau)
        ds = []
        for a_s, b_s, sign in ARM_MOVES:
            a, b = np.searchsorted(t, a_s), np.searchsorted(t, b_s)
            d = integrate(t, a2, a, b)[-1, 1]
            ds.append(d)
            errs.append(d - 0.4 * sign)
        rows[label] = ds
    return np.sqrt(np.mean(np.square(errs))), rows


def main():
    print(" tau(s) | arm vertical rms error (m) | lw raises/lowers ... | rw ...")
    best = None
    for tau in np.round(np.arange(-0.6, 0.61, 0.05), 2):
        e, rows = arm_score(tau)
        if best is None or e < best[0]:
            best = (e, tau)
        print(f" {tau:+.2f}  | {e:.2f} | " + " ".join(f"{v:+.2f}" for v in rows["lw"]) + " | "
              + " ".join(f"{v:+.2f}" for v in rows["rw"]))
    print(f"best tau {best[1]:+.2f} s (rms {best[0]:.2f} m)")
    t, acc, q = load("unity_20261106_200542", "lw")
    for tau in (0.0, best[1]):
        a2 = rerotate(t, acc, q, tau)
        moves = segments(t, a2, q)
        d = [np.linalg.norm(integrate(t, a2, a, b)[-1]) for a, b in moves[1:8]]
        print(f"desk slides/lifts with tau {tau:+.2f}: |d| " + " ".join(f"{v:.2f}" for v in d))


if __name__ == "__main__":
    main()
