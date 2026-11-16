"""Side-by-side overlay of held-out dead-bug clips: source video + DuoMo pseudo-label (left, green)
and + the stream model's prediction (right, red), one MP4 per held-out source video.

The prediction mesh (deploy root R_waist @ R_spine1_pred^T, GT pelvis position) is placed with the
per-frame rigid transform that maps the GT mesh onto DuoMo's world-space fit vertices, then projected
with DuoMo's camera -- the same projection render_overlay.py uses for the pseudo-label.

Run with DuoMo's python (pytorch3d).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/duanyuhan/dyh/motion/data_system/core/flowhmr/tools/datagen")
from render_overlay import label, load_json, project, render_chunk  # noqa: E402

E = Path("/home/duanyuhan/dyh/motion/lines/imudino_deadbug_eval_20261114")
JOBS = Path("/home/duanyuhan/dyh/motion/data_system/runtime/output/web_jobs")
GT_RGB = np.array([90, 200, 120], np.float32)
PRED_RGB = np.array([235, 80, 70], np.float32)
ALPHA = 0.65
SCALE = 0.5
CHUNK = 16


def rigid(src, dst):
    """R, t minimising |R src + t - dst| (Kabsch)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cd - r @ cs


def blend(img, shade, mask, rgb):
    col = shade[..., None] * rgb[::-1][None, None]
    out = img.astype(np.float32)
    out[mask] = (1 - ALPHA) * out[mask] + ALPHA * col[mask]
    return out.astype(np.uint8)


def main():
    report = json.loads((E / "out/deadbug_holdout_report.json").read_text())
    per_seq = {r["cache_index"]: r for r in report["heldout_per_sequence"]}
    index = json.loads((E / "overlay/index.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residuals = []
    by_video = {}
    for item in index:
        by_video.setdefault(item["video"], []).append(item)
    for video, items in by_video.items():
        job_dir = JOBS / items[0]["job"]
        work = job_dir / "input" / "duomo"
        out_dir = work / "out"
        clips = load_json(str(job_dir / "clips.json"))
        clips = clips.get("clips", clips) if isinstance(clips, dict) else clips
        starts = {int(c["index"]): int(c.get("local_start", c["start"])) for c in clips}
        summary = {int(s["index"]): s for s in load_json(str(out_dir / "fit_summary.json"), [])}
        faces = torch.from_numpy(np.load(out_dir / "faces.npy").astype(np.int64))
        cam_pt = out_dir / "preprocess" / "camera.pt"
        k_all = torch.load(cam_pt, map_location="cpu", weights_only=False)["cam_int"].numpy() if cam_pt.exists() else None
        cap = cv2.VideoCapture(str(work / "track.mp4"))
        w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w, h = int(round(w0 * SCALE / 2)) * 2, int(round(h0 * SCALE / 2)) * 2
        out_path = E / "videos" / f"heldout_{Path(video).stem}_gt_vs_pred.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{2 * w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
                               stdin=subprocess.PIPE)
        for item in sorted(items, key=lambda x: x["clip_index"]):
            idx = item["clip_index"]
            data = np.load(E / "overlay" / item["file"])
            fit = np.load(out_dir / f"clip{idx:03d}_fit.npz")["verts"].astype(np.float32)
            fit[..., 1] -= float(summary.get(idx, {}).get("floor_shift_mm") or 0.0) / 1000.0
            src = torch.load(out_dir / f"clip{idx:03d}_duomo.pt", map_location="cpu", weights_only=False)
            cams = np.asarray(src["cam_motion"], np.float32)
            n = min(len(fit), len(cams), len(data["gt"]))
            seq = per_seq[item["cache_index"]]
            cap.set(cv2.CAP_PROP_POS_FRAMES, starts[idx])
            for i0 in range(0, n, CHUNK):
                frames, todo = [], []
                for k in range(i0, min(n, i0 + CHUNK)):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
                    gt = data["gt"][k].astype(np.float32)
                    r, t = rigid(gt, fit[k])
                    residuals.append(float(np.linalg.norm(gt @ r.T + t - fit[k], axis=1).mean()))
                    pred = (data["pred"][k].astype(np.float32) @ r.T + t).astype(np.float32)
                    kmat = k_all[min(starts[idx] + k, len(k_all) - 1)].copy()
                    kmat[0] *= w / w0
                    kmat[1] *= h / h0
                    todo.append(tuple(a.astype(np.float32) for a in project(fit[k], cams[k], kmat)))
                    todo.append(tuple(a.astype(np.float32) for a in project(pred, cams[k], kmat)))
                if not frames:
                    break
                shade, mask = render_chunk(todo[:2 * len(frames)], faces, h, w, device)
                for b, frame in enumerate(frames):
                    k = i0 + b
                    left = blend(frame, shade[2 * b], mask[2 * b], GT_RGB)
                    right = blend(frame, shade[2 * b + 1], mask[2 * b + 1], PRED_RGB)
                    label(left, [f"{video} clip{idx:03d}  pseudo-label (DuoMo)",
                                 f"L thigh {data['gt_L_thigh'][k]:4.0f} knee {data['gt_L_knee'][k]:4.0f}",
                                 f"R thigh {data['gt_R_thigh'][k]:4.0f} knee {data['gt_R_knee'][k]:4.0f}"])
                    label(right, [f"model prediction  (clip thigh err L {seq['L_thigh_abs_err']:.0f} R {seq['R_thigh_abs_err']:.0f} deg)",
                                  f"L thigh {data['pred_L_thigh'][k]:4.0f} knee {data['pred_L_knee'][k]:4.0f}",
                                  f"R thigh {data['pred_R_thigh'][k]:4.0f} knee {data['pred_R_knee'][k]:4.0f}",
                                  "warm-up (excluded)" if 2 * k < 60 else ""])
                    enc.stdin.write(np.concatenate([left, right], axis=1).tobytes())
            print(f"{video} clip{idx:03d} done", flush=True)
        enc.stdin.close()
        enc.wait()
        cap.release()
        print("wrote", out_path, flush=True)
    print("GT-mesh -> DuoMo-fit rigid residual mean/max (mm):",
          round(1000 * float(np.mean(residuals)), 2), round(1000 * float(np.max(residuals)), 2))


if __name__ == "__main__":
    main()
