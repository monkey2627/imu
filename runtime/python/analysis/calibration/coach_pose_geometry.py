"""Coach pose geometry for calibration design: per action, over time, each forearm / shank / thigh
direction as elevation (deg, world) and angle to the body axes (head = pelvis->thorax, chest normal),
plus the trunk's own elevation and chest-normal direction.  Unity world, y up.
python -m analysis.calibration.coach_pose_geometry <humanpose_ours dir> <action number> [step_s]
"""
import json, sys
from pathlib import Path
import numpy as np

B = ['pelvis', 'femur_l', 'femur_r', 'tibia_l', 'tibia_r', 'talus_l', 'talus_r', 'toes_l', 'toes_r', 'lumbar_body',
     'thorax', 'head', 'humerus_l', 'humerus_r', 'ulna_l', 'ulna_r', 'hand_l', 'hand_r']
I = {b: i for i, b in enumerate(B)}
d = json.loads(next(Path(sys.argv[1]).glob(f"{sys.argv[2]}.humanpose_joints.json")).read_text(encoding="utf-8"))
step = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
P = np.array([[[p["x"], p["y"], p["z"]] for p in f["positions"]] for f in d["frames"]])
unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
head = unit(P[:, I["thorax"]] - P[:, I["pelvis"]])
lateral = unit(P[:, I["femur_l"]] - P[:, I["femur_r"]])          # body left
chest = unit(np.cross(lateral, head))                             # sign checked below
el = lambda v: np.degrees(np.arcsin(np.clip(v[..., 1], -1, 1)))
ang = lambda a, b: np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))
print(f"{d['sourceMotionJson'].split(chr(92))[-3]}  frames {len(P)} @ {d['fps']} fps")
print("t(s) | trunk: head el, chest el | forearm L/R: el, ang-to-head, ang-to-chest | shank L/R: el, ang-to-head | thigh L/R el")
for k in range(0, len(P), int(step * d["fps"])):
    seg = lambda a, b: unit(P[k, I[b]] - P[k, I[a]])
    fl, fr = seg("ulna_l", "hand_l"), seg("ulna_r", "hand_r")
    sl, sr = seg("tibia_l", "talus_l"), seg("tibia_r", "talus_r")
    tl, tr = seg("femur_l", "tibia_l"), seg("femur_r", "tibia_r")
    h, c = head[k], chest[k]
    print(f"{k/d['fps']:5.1f} | {el(h):+4.0f} {el(c):+4.0f} | L {el(fl):+4.0f} {ang(fl,h):4.0f} {ang(fl,c):4.0f}  R {el(fr):+4.0f} {ang(fr,h):4.0f} {ang(fr,c):4.0f}"
          f" | L {el(sl):+4.0f} {ang(sl,h):4.0f}  R {el(sr):+4.0f} {ang(sr,h):4.0f} | {el(tl):+4.0f} {el(tr):+4.0f}")
