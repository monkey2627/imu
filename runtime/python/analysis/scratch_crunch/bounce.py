import json, sys, numpy as np, torch, gzip
torch.set_grad_enabled(False); torch.set_num_threads(8)
from production.live_replay import replay_live, _unity_quats_to_smpl
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
import production.mesh_video as mv
f=sys.argv[1]; lo,hi=float(sys.argv[2]),float(sys.argv[3])
rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
live=np.stack([_unity_quats_to_smpl(r["unityLocalQuaternions"]) for r in rows if r.get("type")=="prediction"])
cur=replay_live(f, simulate=True); t=cur.times; k=np.where((t>=lo)&(t<hi))[0]
raw=json.loads(gzip.decompress(mv._get_smpl_model_gzip()).decode("utf-8"))
model={"parents":raw["parents"],"joints":np.asarray(raw["joints"],np.float64).reshape(24,3),"vertices":np.asarray(raw["vertices"],np.float64).reshape(-1,3),
       "bone_ids":np.asarray(raw["boneIds"],np.int64).reshape(-1,4),"weights":np.asarray(raw["weights"],np.float64).reshape(-1,4)}
dom=np.argmax(model["weights"],1); bid=model["bone_ids"][np.arange(len(dom)),dom]
pel=np.isin(bid,[0,1,2])
for name,L in (("live",live),("current",cur.final_local)):
    lift=[]
    for i in k:
        v=mv.skin(model,batch_smpl_matrix_to_unity_quat_xyzw(L[i].astype(np.float32)),"pelvis")
        lift.append(max(0.0, v[pel,1].min()-v[:,1].min()))   # how far lowest-point grounding lifts the buttocks
    lift=np.array(lift)*100
    print(f"{name:8s} {lo}-{hi}s buttocks lifted by lowest-point grounding: median {np.median(lift):.1f} p90 {np.percentile(lift,90):.1f} max {lift.max():.1f} cm; frames >2 cm {np.mean(lift>2)*100:.0f}%")
names={0:"pelvis",1:"Lhip",2:"Rhip",3:"spine1",6:"spine2",9:"spine3",12:"neck",15:"head",13:"Lcollar",14:"Rcollar",16:"Lshoulder",17:"Rshoulder",18:"Lelbow",19:"Relbow",20:"Lwrist",21:"Rwrist",22:"Lhand",23:"Rhand",7:"Lankle",8:"Rankle",10:"Lfoot",11:"Rfoot",4:"Lknee",5:"Rknee"}
from collections import Counter
c=Counter()
for i in k:
    v=mv.skin(model,batch_smpl_matrix_to_unity_quat_xyzw(cur.final_local[i].astype(np.float32)),"pelvis")
    if v[pel,1].min()-v[:,1].min()>0.02: c[names.get(int(bid[np.argmin(v[:,1])]),str(bid[np.argmin(v[:,1])]))]+=1
print("lowest part when buttocks lifted >2 cm:", c.most_common())
