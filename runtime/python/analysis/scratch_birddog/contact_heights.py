import sys, numpy as np, torch
from analysis.birddog_compare_video import replay
from analysis.compare_video import smpl_model
import production.mesh_video as mv
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from production.paths import paths
torch.set_grad_enabled(False)
model, _ = smpl_model()
dom = model["bone_ids"][np.arange(len(model["bone_ids"])), model["weights"].argmax(1)]
REG = {"kneeL": [4], "kneeR": [5], "footL": [7, 10], "footR": [8, 11], "handL": [20, 22], "handR": [21, 23],
       "pelvis": [0, 1, 2]}
masks = {k: np.isin(dom, v) for k, v in REG.items()}
res, diags = replay(paths.recordings / sys.argv[1], rule_on=True)
step = 5
rows = []
for i in range(0, len(res.times), step):
    q = batch_smpl_matrix_to_unity_quat_xyzw(res.final_local[i].astype(np.float32))
    v = mv.skin(model, q, True)
    rows.append([res.times[i]] + [100 * v[m, 1].min() for m in masks.values()]
                + [diags[i]["weights"]["left"] if diags[i] else np.nan, diags[i]["weights"]["right"] if diags[i] else np.nan])
rows = np.array(rows)
np.save(f"analysis/scratch_birddog/heights_{sys.argv[1][:-6]}.npy", rows)
print("  t   | " + " ".join(f"{k:>6}" for k in REG) + " |  wL   wR   (cm above lowest vertex)")
for s in np.arange(0, rows[-1, 0] + 1, 1.0):
    m = (rows[:, 0] >= s) & (rows[:, 0] < s + 1)
    if m.any():
        v = rows[m].mean(0)
        print(f"{s:5.0f} | " + " ".join(f"{x:6.1f}" for x in v[1:8]) + f" | {v[8]:.2f} {v[9]:.2f}")
