import sys, importlib, numpy as np, torch
import production.frame_chain as fc
from analysis.birddog_compare_video import replay
from analysis.compare_video import smpl_model
import production.mesh_video as mv
from production.arm_prior import _globals
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from production.paths import paths
torch.set_grad_enabled(False)
rec = sys.argv[1]; variant = sys.argv[2] if len(sys.argv) > 2 else "prod"
if variant != "prod":
    mod = importlib.import_module(f"analysis.scratch_birddog.{variant}")
    fc.BirdDogContact = mod.BirdDogContact
res, diags = replay(paths.recordings / rec, rule_on=True)
L = res.final_local.astype(np.float64)
G = np.array([_globals(l) for l in L])
model, _ = smpl_model()
pel = []
for i in range(len(L)):
    q = batch_smpl_matrix_to_unity_quat_xyzw(L[i].astype(np.float32))
    v, j = mv.skin(model, q, True, return_joints=True); pel.append(j[0, 1])
pel = np.array(pel) * 100
def ang(a, b): return np.degrees(np.arccos(np.clip((np.trace(a.T @ b, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
d = np.array([[ang(G[i, j], G[i + 1, j]) for j in range(24)] for i in range(len(L) - 1)])
dp = np.abs(np.diff(pel))
mask = res.times[1:] < 59
print(variant, rec, "pelvis |dy| cm: p95 %.2f p99 %.2f max %.1f  n>2cm %d" % (np.percentile(dp[mask], 95), np.percentile(dp[mask], 99), dp[mask].max(), (dp[mask] > 2).sum()))
print("   max joint rot step deg: p99 %.1f  n>15deg %d" % (np.percentile(d[mask].max(1), 99), (d[mask].max(1) > 15).sum()))
names = {0:"pelvis",1:"Lhip",2:"Rhip",4:"Lknee",5:"Rknee",7:"Lankle",8:"Rankle",6:"sp2",9:"sp3",16:"Lsh",17:"Rsh",18:"Lelb",19:"Relb",20:"Lwri",21:"Rwri"}
idx = np.argsort(-dp * mask)[:12]
for i in sorted(idx):
    jt = int(np.argmax(d[i])); dg = diags[i + 1] or {}
    print(f"  t {res.times[i+1]:6.2f} pelvis dy {pel[i+1]-pel[i]:+5.1f} cm  max rot {d[i].max():5.1f} at {names.get(jt, jt)}  spine {dg.get('spineDeg')}  armW {dg.get('armWeights')}")
np.save(f"analysis/scratch_birddog/jump_{variant}_{rec[:-6]}.npy", np.c_[res.times, pel])
