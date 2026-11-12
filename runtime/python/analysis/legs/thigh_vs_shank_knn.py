"""Model-free information check on the coach's Ours captures (10 game actions, 17-IMU labels):
with sensors on forearms + waist + (shanks | thighs), how well do the sensor orientations alone
pin down the legs?  kNN in sensor-orientation space, excluding the query's own +-1.5 s, so a
neighbour must be a *different moment* with similar readings (ambiguity = their leg disagreement).

python -m analysis.legs.thigh_vs_shank_knn <humanpose_ours dir>
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

B = ['pelvis', 'femur_l', 'femur_r', 'tibia_l', 'tibia_r', 'talus_l', 'talus_r', 'toes_l', 'toes_r', 'lumbar_body',
     'thorax', 'head', 'humerus_l', 'humerus_r', 'ulna_l', 'ulna_r', 'hand_l', 'hand_r']
I = {b: i for i, b in enumerate(B)}
# per leg: that leg's own sensor + the waist only (arms are left out: in the coach data they move in
# lock-step with the legs, which a different user's form need not do)
LAYOUTS = {"shank (now)": {"l": ["tibia_l", "pelvis"], "r": ["tibia_r", "pelvis"]},
           "thigh":       {"l": ["femur_l", "pelvis"], "r": ["femur_r", "pelvis"]},
           "thigh+shank": {"l": ["femur_l", "tibia_l", "pelvis"], "r": ["femur_r", "tibia_r", "pelvis"]}}
EXCLUDE = 45   # frames (1.5 s at 30 fps)
POINTING_ONLY = "--full" not in sys.argv


def load(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    rest = Rotation.from_quat([[q["x"], q["y"], q["z"], q["w"]] for q in d["restPose"]["rotations"]])
    rot = np.stack([Rotation.from_quat([[q["x"], q["y"], q["z"], q["w"]] for q in f["rotations"]]).as_matrix()
                    for f in d["frames"]]) @ rest.inv().as_matrix()[None]          # world delta from rest
    pos = np.stack([[[p["x"], p["y"], p["z"]] for p in f["positions"]] for f in d["frames"]])
    # remove the pelvis heading (yaw about +y) so facing direction does not matter
    # yaw maximising tr(Y^T R_pelvis): well defined also with the trunk horizontal (bird dog)
    m = rot[:, I["pelvis"]]
    yaw = np.arctan2(m[:, 0, 2] - m[:, 2, 0], m[:, 0, 0] + m[:, 2, 2])
    c, s = np.cos(-yaw), np.sin(-yaw)
    Y = np.zeros((len(yaw), 3, 3)); Y[:, 0, 0] = c; Y[:, 0, 2] = s; Y[:, 1, 1] = 1; Y[:, 2, 0] = -s; Y[:, 2, 2] = c
    rot = Y[:, None] @ rot
    pos = np.einsum("tij,tkj->tki", Y, pos - pos[:, I["pelvis"]][:, None])
    return d["sourceMotionJson"].replace("\\", "/").split("/")[-3], rot, pos


def knee_angle(pos, side):
    f, t, a = pos[:, I["femur_" + side]], pos[:, I["tibia_" + side]], pos[:, I["talus_" + side]]
    u, v = t - f, a - t
    return np.degrees(np.arccos(np.clip((u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1), -1, 1)))


def main():
    files = sorted(Path(sys.argv[1]).glob("*.humanpose_joints.json"), key=lambda p: int(p.name.split(".")[0]))
    seqs = [load(p) for p in files]
    names = [n for n, _, _ in seqs]
    seq_id = np.concatenate([np.full(len(r), i) for i, (_, r, _) in enumerate(seqs)])
    frame = np.concatenate([np.arange(len(r)) for _, r, _ in seqs])
    rot = np.concatenate([r for _, r, _ in seqs]); pos = np.concatenate([p for _, _, p in seqs])
    knees = {side: knee_angle(pos, side) for side in "lr"}
    print(f"{len(rot)} frames; error = kNN(k=5) leg-joint position (knees+ankles, cm) / knee angle (deg)")
    header = f"{'action':14s}" + "".join(f" | {k:>18s}" for k in LAYOUTS)
    print(header)
    results = {}
    for layout, per_side in LAYOUTS.items():
        err_p, err_k = np.zeros(len(rot)), np.zeros(len(rot))
        for side, bones in per_side.items():
            legs = [I["tibia_" + side], I["talus_" + side]]
            if POINTING_ONLY:   # bone pointing directions (no twist) + the trunk's forward/up axes
                parts = []
                for b in bones:
                    if b == "pelvis":
                        parts += [rot[:, I[b]] @ np.array([0.0, 0.0, 1.0]), rot[:, I[b]] @ np.array([0.0, 1.0, 0.0])]
                    else:
                        child = {"femur": "tibia", "tibia": "talus"}[b.split("_")[0]] + "_" + b.split("_")[1]
                        d = pos[:, I[child]] - pos[:, I[b]]
                        parts.append(d / np.linalg.norm(d, axis=-1, keepdims=True))
                feat = np.concatenate(parts, 1)
            else:
                feat = rot[:, [I[b] for b in bones]].reshape(len(rot), -1)
            _, nn = cKDTree(feat).query(feat, k=300)
            for q in range(len(rot)):
                cand = nn[q][(seq_id[nn[q]] != seq_id[q]) | (np.abs(frame[nn[q]] - frame[q]) > EXCLUDE)][:5]
                err_p[q] += np.linalg.norm(pos[cand][:, legs].mean(0) - pos[q, legs], axis=-1).mean() * 50
                err_k[q] += np.abs(knees[side][cand].mean(0) - knees[side][q]) / 2
        results[layout] = (err_p, err_k)
    for i, n in enumerate(names + ["ALL"]):
        m = seq_id == i if n != "ALL" else np.ones(len(rot), bool)
        print(f"{n:14s}" + "".join(f" | {results[l][0][m].mean():8.1f} cm {results[l][1][m].mean():5.1f}°" for l in LAYOUTS))


if __name__ == "__main__":
    main()
