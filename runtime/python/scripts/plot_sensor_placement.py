#!/usr/bin/env python3
"""Draw the five IMU placements of the current model on the shaded SMPL body (T-pose).

    python scripts/plot_sensor_placement.py --out docs/images/sensor_placement_pnp5.png

Positions are the PNP surface vertices used to synthesise the training data
(SparseIMU/mobileposer/configs/pnp_surface_imu.json).  Filled markers are on the
side of the body that faces the viewer, hollow markers are on the far side.
All views share one scale.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

import mobileposer.articulate as art  # noqa: E402
from mobileposer.config import paths  # noqa: E402

# label, SMPL vertex, colour, text
SENSORS = [
    ("lw", 1961, "#d62728", "L forearm: near wrist,\ndorsal (back of forearm)"),
    ("rw", 5424, "#ff7f0e", "R forearm: near wrist,\ndorsal (back of forearm)"),
    ("ls", 1176, "#1f77b4", "L shank: just below knee,\ninner front (tibia)"),
    ("rs", 4662, "#17becf", "R shank: just below knee,\ninner front (tibia)"),
    ("pelvis", 3021, "#2ca02c", "Pelvis: sacrum\n(back of pelvis)"),
]
E = np.eye(3)
# title, horizontal axis, vertical axis, direction towards the viewer (SMPL: +x body-left, +y up, +z front), labelled sensors
VIEWS = [
    ("Front", E[0], E[1], E[2], {"lw", "rw", "ls", "rs"}),
    ("Left side", -E[2], E[1], E[0], {"ls", "pelvis"}),
    ("Back", -E[0], E[1], -E[2], {"pelvis"}),
    ("Top (front = up)", -E[0], E[2], E[1], {"lw", "rw"}),
]
GAP = 0.9


def face_and_vertex_normals(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    if ((vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12) * (verts - verts.mean(0))).sum(1)).mean() < 0:
        fn, vn = -fn, -vn                                                   # make normals point outwards
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return fn, vn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/images/sensor_placement_pnp5.png"))
    args = p.parse_args()

    model = art.ParametricModel(str(paths.smpl_file), device="cpu")
    pose = torch.eye(3).repeat(24, 1, 1)[None]
    _, _, vert = model.forward_kinematics(pose, torch.zeros(10), torch.zeros(1, 3), calc_mesh=True)
    verts = vert[0].numpy()
    faces = model.face.numpy() if hasattr(model.face, "numpy") else np.asarray(model.face)
    fnorm, vnorm = face_and_vertex_normals(verts, faces)
    y_lo, y_hi = verts[:, 1].min(), verts[:, 1].max()
    y_mid = 0.5 * (y_lo + y_hi)

    fig, ax = plt.subplots(figsize=(19, 8))
    cursor = 0.0
    for title, hvec, vvec, toward, labelled in VIEWS:
        h, v, depth = verts @ hvec, verts @ vvec, verts @ toward
        is_top = title.startswith("Top")
        h_off = cursor - h.min()
        v_off = (y_mid - 0.5 * (v.min() + v.max())) if is_top else 0.0     # centre the top view next to the body
        h, v = h + h_off, v + v_off
        order = np.argsort(depth[faces].mean(1))                            # far faces first
        light = np.clip(fnorm @ toward, 0, 1) * 0.55 + 0.35
        colours = np.stack([light] * 3 + [np.ones_like(light)], axis=1)[order]
        ax.add_collection(PolyCollection(np.stack([h[faces], v[faces]], axis=2)[order], facecolors=colours, edgecolors="none"))
        centre = 0.5 * (h.min() + h.max())
        for k, (label, vertex, colour, text) in enumerate(SENSORS):
            visible = vnorm[vertex] @ toward > 0
            if title == "Left side" and label in ("lw", "rw"):
                continue                                                    # forearms point at the viewer here: no clear projection
            hx, hy = h[vertex], v[vertex]
            ax.scatter([hx], [hy], s=140, facecolors=colour if visible else "none", edgecolors=colour,
                       linewidths=1.2 if visible else 2.4, zorder=5)
            if visible and label in labelled:
                side = 1.0 if hx >= centre else -1.0
                dy = {"lw": 0.10, "rw": 0.10, "ls": -0.05, "rs": 0.05, "pelvis": 0.0}[label] * (1 if not is_top else 2.2)
                ax.annotate(f"{text}\nv{vertex}", (hx, hy), xytext=(hx + side * 0.22, hy + dy), fontsize=9,
                            ha="left" if side > 0 else "right", va="center", color=colour,
                            arrowprops=dict(arrowstyle="-", color=colour, lw=1), zorder=6)
        ax.text(centre, y_hi + 0.12, title, ha="center", fontsize=13)
        cursor = h.max() + GAP
    ax.set_xlim(-0.8, cursor - 0.3)
    ax.set_ylim(y_lo - 0.05, y_hi + 0.25)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Current baseline IMU placement on the SMPL body (T-pose, PNP vertices); filled = facing the viewer, hollow = far side\n"
                 "SlimeVR GUI: Left/Right lower arm, Left/Right ankle (= LOWER_LEG), Hip", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print("saved", args.out)


if __name__ == "__main__":
    main()
