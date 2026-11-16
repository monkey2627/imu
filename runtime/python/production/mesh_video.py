"""Render a Unity JSONL recording's SMPL surface replay to an MP4 (two fixed views, shaded mesh).

Same skinning asset and quaternions as production.mesh_replay (Unity-basis mesh, recorded
unityLocalQuaternions or a re-run through the current chain), rasterised with a painter's algorithm.

    python -m production.mesh_video unity_20261116_022753
    python -m production.mesh_video unity_20261113_232942 --deadbug-fix --model waist_imudino_stream_deadbug_rehabv5
"""
from __future__ import annotations

import argparse
import gzip
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from production.input_recording import resolve_recording_path
from production.web_visualizer import _get_smpl_model_gzip

W, H = 640, 540
_STATE = {}


def _quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz), -1)


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def skin(model, q, floor_contact=False, return_joints=False):
    """Vertices for one frame (24 Unity-local xyzw quaternions), mirroring the WebGL shader.

    The network has no root translation; floor_contact lifts the body so its lowest vertex rests on y=0
    (for lying exercises this shows e.g. the pelvis rising while the shoulders stay down)."""
    parents, joints = model["parents"], model["joints"]
    G, P = [None] * 24, [None] * 24
    mats = np.zeros((24, 3, 4))
    for j in range(24):
        a = parents[j]
        if a < 0:
            G[j], P[j] = q[j], np.zeros(3)
        else:
            G[j] = _quat_mul(G[a], q[j])
            P[j] = P[a] + _quat_mat(G[a]) @ (joints[j] - joints[a])
        R = _quat_mat(G[j])
        mats[j, :, :3], mats[j, :, 3] = R, P[j] - R @ joints[j]
    blend = np.einsum("vk,vkab->vab", model["weights"], mats[model["bone_ids"]])
    verts = np.einsum("vab,vb->va", blend[:, :, :3], model["vertices"]) + blend[:, :, 3]
    joints = np.array(P)
    if floor_contact:
        lowest_offset = -float(verts[:, 1].min())
        offset = lowest_offset
        if floor_contact == "pelvis":
            chest_up = float(_quat_mat(q[0])[1, 2])
            supine = float(np.clip((chest_up - 0.45) / 0.25, 0.0, 1.0))
            pelvis_offset = 0.153 * chest_up
            offset = (1.0 - supine) * lowest_offset + supine * pelvis_offset
        verts[:, 1] += offset
        joints[:, 1] += offset
    return (verts, joints) if return_joints else verts


def _view(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rx @ ry


def _render(verts, faces, view, center, scale, floor_y):
    img = np.full((H, W, 3), (34, 27, 23), np.uint8)
    v = (verts - center) @ view.T
    xy = np.stack((W / 2 + v[:, 0] * scale, H * 0.55 - v[:, 1] * scale), -1)
    if floor_y is not None:   # floor grid at world y = floor_y around the body
        for a in np.linspace(-1.2, 1.2, 9):
            for seg in (((a, -1.2), (a, 1.2)), ((-1.2, a), (1.2, a))):
                w = np.array([[center[0] + x, floor_y, center[2] + z] for x, z in seg])
                f = (w - center) @ view.T
                pts = [(int(W / 2 + u[0] * scale), int(H * 0.55 - u[1] * scale)) for u in f]
                cv2.line(img, pts[0], pts[1], (70, 64, 58), 1, cv2.LINE_AA)
    tri = v[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    visible = normal[:, 2] > 0
    light = np.array([-0.3, 0.6, 0.75]); light /= np.linalg.norm(light)
    shade = 0.32 + 0.68 * np.clip(normal @ light, 0, 1)
    order = np.argsort(tri[:, :, 2].mean(1))
    pts = np.round(xy[faces] * 4).astype(np.int32)
    base = np.array([219, 168, 92.0])   # BGR (light blue mesh)
    for k in order:
        if visible[k]:
            cv2.fillConvexPoly(img, pts[k], (base * shade[k]).tolist(), cv2.LINE_AA, 2)
    return img


def _init(state):
    _STATE.update(state)


def _frame(index):
    s = _STATE
    verts, joints = skin(s["model"], s["quats"][index], s["ground"], True)
    panels = []
    for name, view in s["views"]:
        img = _render(verts, s["faces"], view, s["center"], s["scale"], s["floor"])
        cv2.putText(img, name, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        panels.append(img)
    out = np.concatenate(panels, 1)
    torso = joints[12] - joints[0]   # pelvis -> neck; >0 means the neck is higher than the pelvis
    pelvis_h = joints[0, 1] - (s["floor"] or 0.0)
    cv2.putText(out, f"torso (pelvis->neck) {np.degrees(np.arcsin(torso[1] / np.linalg.norm(torso))):+5.1f} deg   "
                f"pelvis height {100 * pelvis_h:4.1f} cm", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 1,
                cv2.LINE_AA)
    cv2.putText(out, f"{s['title']}   t = {index / 60:6.2f} s", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)
    return out


def load_quaternions(recording: Path, deadbug_fix: bool, baseline: bool, model_id: str | None):
    from production.mesh_replay import deadbug_fixed_quaternions, recorded_unity_quaternions
    rows = [json.loads(line) for line in recording.read_text(encoding="utf-8").splitlines() if line.strip()]
    frames = sum(int(r.get("ticks", 1)) for r in rows if r.get("type") == "tick")
    if deadbug_fix or baseline:
        q, _ = deadbug_fixed_quaternions(recording, rows, frames, "off" if baseline else "auto", model_id)
        return q, model_id or rows[0].get("model", "")
    return recorded_unity_quaternions(rows, frames), rows[0].get("model", "")


def write_video(recording: Path, output: Path, quats: np.ndarray, title: str, fps: int = 30, workers: int = 1,
                floor_contact: bool = True):
    raw = json.loads(gzip.decompress(_get_smpl_model_gzip()).decode("utf-8"))
    model = {"parents": raw["parents"], "joints": np.asarray(raw["joints"], np.float64).reshape(24, 3),
             "vertices": np.asarray(raw["vertices"], np.float64).reshape(-1, 3),
             "bone_ids": np.asarray(raw["boneIds"], np.int64).reshape(-1, 4),
             "weights": np.asarray(raw["weights"], np.float64).reshape(-1, 4)}
    faces = np.asarray(raw["faces"], np.int64).reshape(-1, 3)
    step = max(1, round(60 / fps))
    indices = list(range(0, len(quats), step))
    # Framing from a coarse pass: body centre, extent, floor height and the body's lateral axis.
    sample = np.stack([skin(model, quats[i], floor_contact) for i in indices[::max(1, len(indices) // 40)]])
    center = np.median(sample.reshape(-1, 3), 0)
    extent = np.percentile(np.abs(sample - center).reshape(-1, 3), 99.5, axis=0).max()
    floor = 0.0 if floor_contact else None
    # Side view looks along the mean hip axis (left hip -> right hip), projected on the floor.
    lateral = np.zeros(3)
    for i in indices[::max(1, len(indices) // 40)]:
        q = quats[i][0]
        lateral += _quat_mat(q) @ np.array([1.0, 0, 0])
    lateral[1] = 0
    yaw_side = float(np.arctan2(lateral[0], lateral[2])) if np.linalg.norm(lateral) > 1e-6 else 0.0
    ground_label = "pelvis contact" if floor_contact == "pelvis" else "lowest point on floor"
    views = [("side" + (f" ({ground_label})" if floor_contact else " (pelvis fixed)"), _view(yaw_side + np.pi, 0.0)), ("3/4 view", _view(yaw_side + np.pi + 0.9, 0.35))]
    scale = 0.42 * min(W, H) / max(extent, 0.3)
    state = dict(model=model, faces=faces, quats=quats, views=views, center=center, scale=scale, floor=floor,
                 ground=floor_contact,
                 title=title)
    output = output.with_suffix(".mp4")
    writer = None
    for codec in ("mp4v", "avc1"):   # MPEG-4 part 2 is broadly available; H.264 is optional
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, (2 * W, H))
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"could not open an MP4 writer for {output}")
    if workers <= 1:
        _init(state)
        rendered = map(_frame, indices)
        for n, img in enumerate(rendered):
            writer.write(img)
            if n % 300 == 0:
                print(f"frame {n}/{len(indices)}", flush=True)
    else:
        with Pool(workers, initializer=_init, initargs=(state,)) as pool:
            for n, img in enumerate(pool.imap(_frame, indices, chunksize=4)):
                writer.write(img)
                if n % 300 == 0:
                    print(f"frame {n}/{len(indices)}", flush=True)
    writer.release()
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--deadbug-fix", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1,
                    help="number of frame-render workers (use 1 on restricted Windows environments)")
    ap.add_argument("--pelvis-fixed", action="store_true", help="no floor contact: keep the pelvis at the origin")
    a = ap.parse_args()
    src = resolve_recording_path(a.recording)
    quats, model_id = load_quaternions(src, a.deadbug_fix, a.baseline, a.model)
    dst = a.out or src.with_name(src.stem + "_smpl.mp4")
    print(write_video(src, dst, quats, f"{src.stem}  {model_id}", a.fps, workers=max(1, a.workers),
                      # In supine dead-bug playback, the lowest vertex is often a foot.
                      # Re-anchoring to it would lift the torso whenever a leg extends.
                      floor_contact=("pelvis" if "curlup" in (model_id or "").lower() else
                                     not a.pelvis_fixed and "deadbug" not in (model_id or "").lower())))
