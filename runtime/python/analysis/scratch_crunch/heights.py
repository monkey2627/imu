import json, sys, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from production.live_replay import replay_live, _unity_quats_to_smpl
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
import production.mesh_video as mv
f=sys.argv[1]; lo,hi=float(sys.argv[2]),float(sys.argv[3]); step=float(sys.argv[4])
rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
live=np.stack([_unity_quats_to_smpl(r["unityLocalQuaternions"]) for r in rows if r.get("type")=="prediction"])
cur=replay_live(f, simulate=True); t=cur.times
import gzip
raw=json.loads(gzip.decompress(mv._get_smpl_model_gzip()).decode("utf-8"))
model={"parents":raw["parents"],"joints":np.asarray(raw["joints"],np.float64).reshape(24,3),"vertices":np.asarray(raw["vertices"],np.float64).reshape(-1,3),
       "bone_ids":np.asarray(raw["boneIds"],np.int64).reshape(-1,4),"weights":np.asarray(raw["weights"],np.float64).reshape(-1,4)}
# vertex groups by dominant bone
dom=np.argmax(model["weights"],1); bid=model["bone_ids"][np.arange(len(dom)),dom]
groups={"pelvis":[0,1,2],"feet":[7,8,10,11],"shank":[4,5],"upperback":[6,9,13,14,16,17],"head":[12,15]}
last=-9
for i,tt in enumerate(t):
    if tt<lo or tt>hi or tt-last<step: continue
    last=tt; s=f"{tt:5.1f}"
    for name,L in (("live",live),("cur",cur.final_local)):
        q=batch_smpl_matrix_to_unity_quat_xyzw(L[i].astype(np.float32))
        v=mv.skin(model,q,"pelvis")
        hs={k:v[np.isin(bid,b),1].min() for k,b in groups.items()}
        s+=f" | {name} " + " ".join(f"{k} {hs[k]*100:4.0f}" for k in ("pelvis","feet","upperback","head"))
    print(s)
