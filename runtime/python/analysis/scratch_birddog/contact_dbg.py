import sys, numpy as np, torch
from analysis.birddog_compare_video import replay
from analysis.compare_video import smpl_model
import production.mesh_video as mv
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from production.paths import paths
torch.set_grad_enabled(False)
model, _ = smpl_model()
dom = model["bone_ids"][np.arange(len(model["bone_ids"])), model["weights"].argmax(1)]
kneeM = {4: np.isin(dom, [4]), 5: np.isin(dom, [5])}; handM = {20: np.isin(dom, [20, 22]), 21: np.isin(dom, [21, 23])}
res, diags = replay(paths.recordings / sys.argv[1], rule_on=True)
for t in (2, 5, 10, 12, 28, 31, 36, 58):
    i = int(np.argmin(abs(res.times - t)))
    q = batch_smpl_matrix_to_unity_quat_xyzw(res.final_local[i].astype(np.float32))
    v, j = mv.skin(model, q, True, return_joints=True)
    d = diags[i]
    print(f"t {t}: spine {d['spineDeg']}  armW {d['armWeights']}  legW {d['weights']}  forearm {d['sensorElevationDeg']['leftForearmDeg']},{d['sensorElevationDeg']['rightForearmDeg']}")
    print("   knee joint-surface cm", [round(100*(j[k,1]-v[m,1].min()),1) for k,m in kneeM.items()],
          " wrist joint-surface cm", [round(100*(j[k,1]-v[m,1].min()),1) for k,m in handM.items()],
          " knee y", [round(100*j[k,1],1) for k in (4,5)], " wrist y", [round(100*j[k,1],1) for k in (20,21)])
