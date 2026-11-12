"""Hypothesis: the firmware removes gravity with a LAGGED orientation, so tilting leaks
    a_meas_world = a_true + g * (R(t) R(t - tau)^T - I) up        (up = +y, g = 9.81; sign s fitted)
into the linear acceleration.  Pure slides and turns about vertical do not change the tilt (no leak);
arm raises / shank lifts do.  Fit tau (and s) on the desk test's tilt moves, then remove the predicted
leak from (1) the desk test and (2) the arm recording unity_20261103_002526, and re-integrate.

python -m analysis.acceleration.acc_gravity_lag
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from production.input_recording import resolve_recording_path

G = 9.81
UP = np.array([0.0, 1.0, 0.0])


def load(name, label):
    rows = [json.loads(l) for l in resolve_recording_path(name).read_text(encoding="utf-8").splitlines() if l.strip()]
    ticks = [r for r in rows if r.get("type") == "tick"]
    t = np.array([r["t"] for r in ticks]); t -= t[0]
    s = [r["samples"][label] for r in ticks]
    acc = np.array([x["reference_adjusted_accel_xyz"] for x in s], float)
    q = np.array([x["quat_xyzw"] for x in s], float)
    q[np.sum(q * np.roll(q, 1, 0), 1) < 0] *= -1
    return t, acc, q


def leak(t, q, tau):
    """g * (R(t) R(t-tau)^T - I) up for every sample (orientations slerped at t - tau)."""
    from scipy.spatial.transform import Slerp
    R = Rotation.from_quat(q)
    lagged = Slerp(t, R)(np.clip(t - tau, t[0], t[-1]))
    return G * ((R * lagged.inv()).apply(UP) - UP)


def integrate(t, acc, a, b):
    pre, post = slice(max(0, a - 20), a), slice(b, min(len(t), b + 20))
    tt = t[a:b]
    x = acc[a:b] - (acc[pre].mean(0) + np.outer((tt - tt[0]) / (tt[-1] - tt[0]), acc[post].mean(0) - acc[pre].mean(0)))
    v = np.vstack([np.zeros(3), np.cumsum((x[1:] + x[:-1]) / 2 * np.diff(tt)[:, None], 0)])
    v -= np.outer((tt - tt[0]) / (tt[-1] - tt[0]), v[-1])
    p = np.vstack([np.zeros(3), np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(tt)[:, None], 0)])
    return p


def segments(t, acc, q, rate_thr=8, acc_thr=0.35):
    R = Rotation.from_quat(q)
    rate = np.r_[0, np.degrees((R[:-1].inv() * R[1:]).magnitude()) / np.maximum(np.diff(t), 1e-3)]
    raw = (rate < rate_thr) & (np.linalg.norm(acc, axis=1) < acc_thr)
    still = np.zeros(len(t), bool); i = 0
    while i < len(t):
        if raw[i]:
            j = i
            while j < len(t) and raw[j]:
                j += 1
            if t[j - 1] - t[i] >= 0.4:
                still[i:j] = True
            i = j
        else:
            i += 1
    out, i = [], 0
    while i < len(t):
        if still[i]:
            i += 1; continue
        j = i
        while j < len(t) and not still[j]:
            j += 1
        if t[j - 1] - t[i] > 0.25 and i > 20 and j < len(t) - 20:
            out.append((i, j))
        i = j
    return out


def main():
    t, acc, q = load("unity_20261106_194609", "rw")
    moves = segments(t, acc, q)
    tilt = [m for k, m in enumerate(moves) if k >= 16]                # the six tilts
    sel = np.concatenate([np.arange(a, b) for a, b in tilt])
    best = None
    for tau in np.arange(0.0, 0.61, 0.02):
        L = leak(t, q, tau)
        for sign in (1, -1):
            r = acc[sel] - sign * L[sel]
            e = np.sqrt((r ** 2).sum(1).mean())
            if best is None or e < best[0]:
                best = (e, tau, sign)
    e0 = np.sqrt((acc[sel] ** 2).sum(1).mean())
    print(f"tilt moves: acc rms {e0:.2f} m/s^2 -> {best[0]:.2f} after removing the leak, tau {best[1]:.2f} s, sign {best[2]:+d}")
    for tau_show in (0.0, 0.1, 0.2, 0.3, 0.4):
        r = acc[sel] - best[2] * leak(t, q, tau_show)[sel]
        print(f"   tau {tau_show:.1f}: residual rms {np.sqrt((r ** 2).sum(1).mean()):.2f}")
    tau, sign = best[1], best[2]
    fixed = acc - sign * leak(t, q, tau)
    print("\ndesk test, displacement |d| raw -> leak-corrected (slides expect ~0.5 m; lifts: peak height)")
    for k, (a, b) in enumerate(moves):
        p0, p1 = integrate(t, acc, a, b), integrate(t, fixed, a, b)
        if k <= 5:
            print(f"  slide {k}: {np.linalg.norm(p0[-1]):.2f} -> {np.linalg.norm(p1[-1]):.2f}")
        elif 7 <= k <= 9:
            print(f"  lift  {k}: peak height {p0[:, 1].max():+.2f}/{p0[:, 1].min():+.2f} -> {p1[:, 1].max():+.2f}/{p1[:, 1].min():+.2f} m"
                  f" (end {p0[-1, 1]:+.2f} -> {p1[-1, 1]:+.2f})")
    # arm recording: expected forearm sensor rise ~0.4 m on raises, -0.4 m on lowers
    print("\narm recording unity_20261103_002526: vertical displacement raw -> leak-corrected (expect +0.4 / -0.4)")
    for label in ("lw", "rw"):
        t2, acc2, q2 = load("unity_20261103_002526", label)
        fixed2 = acc2 - sign * leak(t2, q2, tau)
        line = f"  {label}:"
        for a_s, b_s, exp in ((2.6, 4.1, +1), (5.2, 7.1, -1), (13.6, 14.8, +1), (15.9, 17.2, -1), (22.1, 23.1, +1), (24.3, 25.4, -1)):
            a, b = np.searchsorted(t2, a_s), np.searchsorted(t2, b_s)
            d0, d1 = integrate(t2, acc2, a, b)[-1, 1], integrate(t2, fixed2, a, b)[-1, 1]
            line += f"  {'+' if exp > 0 else '-'}: {d0:+.2f}->{d1:+.2f}"
        print(line)


if __name__ == "__main__":
    main()
