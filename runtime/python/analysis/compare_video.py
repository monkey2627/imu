"""Multi-column SMPL mesh comparison video (shared by the per-exercise comparison scripts).

Each column is one variant's (P, 24, 3, 3) SMPL local rotations on the same 30 Hz
prediction timeline.  Top row: side view, bottom row: 3/4 view; every column uses the
same camera, and each body's lowest vertex rests on the floor (pose-only network).
Per-frame overlay text is computed by the caller in the main process.
"""
from __future__ import annotations

import gzip
import json
import math
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import production.mesh_video as mv
from production.arm_prior import PARENTS, _globals
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from production.web_visualizer import _get_smpl_model_gzip

PW, PH = 400, 340
HEADER = 70
FONT = "C:/Windows/Fonts/msyh.ttc"
_S = {}


def smpl_model():
    raw = json.loads(gzip.decompress(_get_smpl_model_gzip()).decode("utf-8"))
    model = {"parents": raw["parents"], "joints": np.asarray(raw["joints"], np.float64).reshape(24, 3),
             "vertices": np.asarray(raw["vertices"], np.float64).reshape(-1, 3),
             "bone_ids": np.asarray(raw["boneIds"], np.int64).reshape(-1, 4),
             "weights": np.asarray(raw["weights"], np.float64).reshape(-1, 4)}
    return model, np.asarray(raw["faces"], np.int64).reshape(-1, 3)


def fk(local, offsets):
    """Global rotations (24,3,3) and pelvis-relative joint positions (24,3) in SMPL axes."""
    g = _globals(np.asarray(local, dtype=np.float64))
    p = np.zeros((24, 3))
    for j in range(1, 24):
        p[j] = p[PARENTS[j]] + g[PARENTS[j]] @ offsets[j]
    return np.asarray(g), p


def elevation(g, joint, child, offsets):
    v = g[joint] @ offsets[child]
    return math.degrees(math.asin(float(np.clip(v[1] / np.linalg.norm(v), -1, 1))))


def angle_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))


def _init(state):
    mv.W, mv.H = PW, PH          # worker processes re-import mesh_video with its default size
    _S.update(state)


def _frame(index):
    s = _S
    rows = []
    for view_index, view in enumerate(s["views"]):
        panels = []
        for key in s["order"]:
            verts = mv.skin(s["model"], s["quats"][key][index], True)
            img = mv._render(verts, s["faces"], view, s["center"], s["scale"], 0.0)
            if view_index == 0:
                for line, text in enumerate(s["text"][key][index]):
                    cv2.putText(img, text, (8, 20 + 18 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 255), 1,
                                cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (PW - 1, PH - 1), (60, 60, 60), 1)
            panels.append(img)
        rows.append(np.concatenate(panels, 1))
    out = np.concatenate([s["header"]] + rows, 0)
    cv2.putText(out, f"t = {s['times'][index]:6.2f} s", (out.shape[1] - 170, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def _header(order, titles, subtitle):
    img = Image.new("RGB", (PW * len(order), HEADER), (23, 27, 34))
    draw = ImageDraw.Draw(img)
    big, small = ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT, 13)
    for column, key in enumerate(order):
        draw.text((column * PW + 10, 8), titles[key], font=big, fill=(235, 235, 235))
    draw.text((10, 44), subtitle, font=small, fill=(150, 160, 175))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def render(out: Path, columns, times, text, subtitle, frame_key=None, side_yaw_offset=0.0, workers=8,
           yaw_axis=(1.0, 0.0, 0.0)):
    """columns: list of (key, title, locals (P,24,3,3)); text: {key: [lines per frame]}.
    yaw_axis: pelvis axis the first view looks along (default lateral; (0,0,1) = chest, for side lying)."""
    model, faces = smpl_model()
    order = [key for key, _t, _l in columns]
    titles = {key: title for key, title, _l in columns}
    quats = {key: np.stack([batch_smpl_matrix_to_unity_quat_xyzw(pose) for pose in np.asarray(local, np.float32)])
             for key, _t, local in columns}
    n = min(len(q) for q in quats.values())
    frame_key = frame_key or order[-1]
    sample_idx = list(range(0, n, max(1, n // 40)))
    sample = np.stack([mv.skin(model, quats[frame_key][i], True) for i in sample_idx])
    center = np.median(sample.reshape(-1, 3), 0)
    extent = np.percentile(np.abs(sample - center).reshape(-1, 3), 99.5, axis=0).max()
    lateral = sum(mv._quat_mat(quats[frame_key][i][0]) @ np.asarray(yaw_axis, float) for i in sample_idx)
    lateral[1] = 0
    yaw = float(np.arctan2(lateral[0], lateral[2])) + side_yaw_offset
    views = [mv._view(yaw + np.pi, 0.0), mv._view(yaw + np.pi + 0.9, 0.35)]
    mv.W, mv.H = PW, PH
    state = dict(model=model, faces=faces, quats=quats, order=order, views=views, center=center,
                 scale=0.42 * min(PW, PH) / max(extent, 0.3), text=text, times=times,
                 header=_header(order, titles, subtitle))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (PW * len(order), HEADER + 2 * PH))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open {out}")
    # Some embedded/managed Python environments (including the mobileposer
    # runtime on Windows) disallow creating child processes.  Keep a
    # deterministic in-process fallback for recorded-video rendering.
    if workers <= 1:
        _init(state)
        frames = enumerate(map(_frame, range(n)))
    else:
        pool = Pool(workers, initializer=_init, initargs=(state,))
        frames = enumerate(pool.imap(_frame, range(n), chunksize=4))
    try:
        for k, img in frames:
            writer.write(img)
            if k % 200 == 0:
                print(f"  frame {k}/{n}", flush=True)
    finally:
        if workers > 1:
            pool.close()
            pool.join()
    writer.release()
    return out
