"""Freeze the wrist/ankle five-IMU attachment config (2026-11-12).

Same ray-intersection construction as mobileposer.surface_imu.make_config, with
the forearm/shank sensors moved from 75%/88% of the bone to the distal end
(wrist/ankle strap, still rigid with forearm joint 18/19 and shank joint 4/5,
same lateral surface direction). The waist attachment is copied verbatim from
no_head_surface_imu.json so it stays bit-identical.

Run inside a MobilePoser workspace (PYTHONPATH=<ws>):
    python make_wristankle_config.py --base <ws>/mobileposer/configs/no_head_surface_imu.json \
        --out <ws>/mobileposer/configs/wristankle_surface_imu.json --fraction 0.95
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.surface_imu import file_hash, load_attachments


def attach(model, j, v, bone, origin, direction, description, fraction):
    tri = v[model.face]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(direction, e2)
    det = (e1 * h).sum(1)
    safe = np.where(abs(det) > 1e-10, det, 1.0)
    s = origin - tri[:, 0]
    u = (s * h).sum(1) / safe
    q = np.cross(s, e1)
    b = (q * direction).sum(1) / safe
    t = (e2 * q).sum(1) / safe
    eligible = (abs(det) > 1e-10) & (u >= 0) & (b >= 0) & (u + b <= 1) & (t > 0) & (t < 0.25)
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError(f"No surface intersection for bone {bone}")
    face = int(candidates[np.argmin(t[candidates])])
    normal = np.cross(e1[face], e2[face])
    normal /= np.linalg.norm(normal)
    if normal @ direction < 0:
        normal = -normal
    tangent = np.array([0.0, 1.0, 0.0])
    if abs(tangent @ normal) > 0.9:
        tangent = np.array([1.0, 0.0, 0.0])
    tangent -= normal * (normal @ tangent)
    tangent /= np.linalg.norm(tangent)
    mounting = np.stack((tangent, np.cross(normal, tangent), normal), axis=1)
    return dict(joint=bone, face=face, vertices=list(map(int, model.face[face])),
                weights=[float(1 - u[face] - b[face]), float(u[face]), float(b[face])],
                mount_rotation=mounting.tolist(), description=description, bone_fraction=fraction,
                neutral_position_m=(origin + t[face] * direction).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fraction", type=float, default=0.95)
    args = ap.parse_args()
    base = json.loads(args.base.read_text())
    model = ParametricModel(paths.smpl_file)
    joint, vertex = model.get_zero_pose_joint_and_vertex(torch.zeros(10))
    j, v = joint[0].numpy(), vertex[0].numpy()
    attachments = {}
    for side, offset, sign in (("l", 0, 1), ("r", 1, -1)):
        for label, bone, child, direction, description in (
            ("w", 18, 20, [0, 1, 0], "distal forearm lateral surface at the wrist strap"),
            ("s", 4, 7, [sign, 0, 0], "lateral distal shank at the ankle strap"),
        ):
            bone, child = bone + offset, child + offset
            f = args.fraction
            attachments[side + label] = attach(model, j, v, bone, (1 - f) * j[bone] + f * j[child],
                                               np.array(direction, float), description, f)
    attachments["waist"] = base["attachments"]["waist"]
    config = dict(version=1, smpl_sha256=file_hash(paths.smpl_file),
                  reference="wrist/ankle straps on forearm/shank distal end; waist unchanged",
                  orientation="bone_calibrated", acceleration="world_linear_m_s2", pose_blendshape=False,
                  attachments=attachments)
    with args.out.open("x") as f:
        json.dump(config, f, indent=2)
    load_attachments(args.out)
    for name, a in attachments.items():
        old = base["attachments"].get(name)
        moved = np.linalg.norm(np.subtract(a["neutral_position_m"], old["neutral_position_m"])) if old else 0
        print(name, a["joint"], a["vertices"], a.get("bone_fraction"), f"moved {moved * 100:.1f} cm")
    print("sha256", file_hash(args.out))


if __name__ == "__main__":
    main()
