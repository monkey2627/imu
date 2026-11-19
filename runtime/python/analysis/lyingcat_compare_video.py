"""Side-by-side SMPL mesh video of the lying-cat contact rule variants on one recording.

Columns: raw network output | original rule | spine-only rewrite | + pelvis stabilisation |
current production rule (+ palms fixed fore-aft relative to the knees).
Top row: side view; bottom row: 3/4 view.  Same camera for every column, each body's
lowest vertex on the floor (the network has no root translation).

    python -m analysis.lyingcat_compare_video [recording] [--workers 8]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import production.frame_chain as frame_chain
import production.mesh_video as mv
from production.arm_prior import _globals
from production.deadbug_replay import replay
from production.model_backends import ACTION_RUNTIME_MODELS
from production.paths import paths
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from production.web_visualizer import _get_smpl_model_gzip

PW, PH = 400, 340          # one panel
HEADER = 70
FONT = "C:/Windows/Fonts/msyh.ttc"
_S = {}


def _variants():
    from analysis.lyingcat_variants.v1_original import LyingCatContact as V1
    from analysis.lyingcat_variants.v2_spine_only import LyingCatContact as V2
    from analysis.lyingcat_variants.v3_pelvis import LyingCatContact as V3
    from analysis.lyingcat_variants.v4_foreaft import LyingCatContact as V4
    return [
        ("network", "① 网络原始输出（无规则）", None),
        ("v1", "② 旧规则：前臂竖直，斜腿凑平面", V1),
        ("v2", "③ 大腿竖直、肘直，上段脊柱补高度", V2),
        ("v3", "④ 当前：③＋骨盆稳定（腰椎弯曲）", V3),
        ("v4", "⑤ 试验（未采用）：④＋手前后固定", V4),
    ]


def _run(recording, model_id):
    tracks, offsets = {}, None
    for key, _title, cls in _variants():
        if cls is None:
            continue
        frame_chain.LyingCatContact = cls
        result = replay(paths.recordings / recording, model_id=model_id,
                        start_context_frames=60, start_offset_frames=60)
        tracks[key] = result.final_local
        if key == "v4":
            tracks["network"] = result.raw_local
            offsets = cls().offsets
            times = result.represented_frames / 60.0
    return tracks, offsets, times


def _metrics(local, offsets):
    g = _globals(local.astype(np.float64))
    p = np.zeros((24, 3))
    for j in range(1, 24):
        from production.arm_prior import PARENTS
        p[j] = p[PARENTS[j]] + g[PARENTS[j]] @ offsets[j]

    def elev(j, c):
        v = g[j] @ offsets[c]
        return math.degrees(math.asin(float(np.clip(v[1] / np.linalg.norm(v), -1, 1))))
    knee = 0.5 * (p[4] + p[5])
    head = p[12] - p[0]
    head[1] = 0.0
    head /= max(np.linalg.norm(head), 1e-9)
    hand = 0.5 * (p[20] + p[21])
    return (f"hand-knee height L {100 * (p[20, 1] - knee[1]):+3.0f} R {100 * (p[21, 1] - knee[1]):+3.0f} cm",
            f"hand-knee fore-aft {100 * float((hand - knee) @ head):4.0f} cm",
            f"thigh L {elev(1, 4):+3.0f} R {elev(2, 5):+3.0f}  pelvis {elev(0, 3):+3.0f} deg")


def _init(state):
    mv.W, mv.H = PW, PH          # worker processes re-import mesh_video with its default size
    _S.update(state)


def _frame(index):
    s = _S
    rows = []
    for view_index, (_vname, view) in enumerate(s["views"]):
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
    body = np.concatenate(rows, 0)
    out = np.concatenate([s["header"], body], 0)
    cv2.putText(out, f"t = {s['times'][index]:6.2f} s", (out.shape[1] - 170, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def _header(order, titles, subtitle):
    width = PW * len(order)
    img = Image.new("RGB", (width, HEADER), (23, 27, 34))
    draw = ImageDraw.Draw(img)
    big, small = ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT, 13)
    for column, key in enumerate(order):
        draw.text((column * PW + 10, 8), titles[key], font=big, fill=(235, 235, 235))
    draw.text((10, 44), subtitle, font=small, fill=(150, 160, 175))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", nargs="?", default="unity_20261116_195301.jsonl")
    parser.add_argument("--model", default=ACTION_RUNTIME_MODELS["lyingcat"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)

    tracks, offsets, times = _run(args.recording, args.model)
    order = [key for key, _t, _c in _variants()]
    titles = {key: title for key, title, _c in _variants()}
    raw = json.loads(gzip.decompress(_get_smpl_model_gzip()).decode("utf-8"))
    model = {"parents": raw["parents"], "joints": np.asarray(raw["joints"], np.float64).reshape(24, 3),
             "vertices": np.asarray(raw["vertices"], np.float64).reshape(-1, 3),
             "bone_ids": np.asarray(raw["boneIds"], np.int64).reshape(-1, 4),
             "weights": np.asarray(raw["weights"], np.float64).reshape(-1, 4)}
    faces = np.asarray(raw["faces"], np.int64).reshape(-1, 3)
    quats = {key: np.stack([batch_smpl_matrix_to_unity_quat_xyzw(pose) for pose in tracks[key].astype(np.float32)])
             for key in order}
    text = {key: [_metrics(local, offsets) for local in tracks[key]] for key in order}
    n = min(len(q) for q in quats.values())

    # Shared framing from the current-rule track; side view along the mean hip axis.
    sample_idx = list(range(0, n, max(1, n // 40)))
    sample = np.stack([mv.skin(model, quats["v4"][i], True) for i in sample_idx])
    center = np.median(sample.reshape(-1, 3), 0)
    extent = np.percentile(np.abs(sample - center).reshape(-1, 3), 99.5, axis=0).max()
    lateral = sum(mv._quat_mat(quats["v4"][i][0]) @ np.array([1.0, 0, 0]) for i in sample_idx)
    lateral[1] = 0
    yaw = float(np.arctan2(lateral[0], lateral[2]))
    views = [("side", mv._view(yaw + np.pi, 0.0)), ("3/4", mv._view(yaw + np.pi + 0.9, 0.35))]
    mv.W, mv.H = PW, PH          # mv._render reads the panel size from module globals
    scale = 0.42 * min(PW, PH) / max(extent, 0.3)
    subtitle = (f"{args.recording} · 模型 {args.model} · 上排侧视 / 下排 3/4 视角 · 相机统一、"
                "最低点落地 · 青色数字：手相对膝的高度（0=同一平面）、手到膝前后距离（应保持不变）、大腿仰角（-90=竖直）、骨盆俯仰")
    header = _header(order, titles, subtitle)
    state = dict(model=model, faces=faces, quats=quats, order=order, views=views, center=center,
                 scale=scale, text=text, times=times, header=header)

    out = args.out or (paths.recordings / "action_eval_20261119" /
                       f"lyingcat_compare_{Path(args.recording).stem}.mp4")
    size = (PW * len(order), HEADER + 2 * PH)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open {out}")
    with Pool(args.workers, initializer=_init, initargs=(state,)) as pool:
        for k, img in enumerate(pool.imap(_frame, range(n), chunksize=4)):
            writer.write(img)
            if k % 100 == 0:
                print(f"frame {k}/{n}", flush=True)
    writer.release()
    print(out)


if __name__ == "__main__":
    main()
