import json, sys
import numpy as np, torch
from analysis.compare_video import fk
from production.live_replay import _unity_quats_to_smpl, replay_live
from production.lyingcat_contact import LyingCatContact
torch.set_grad_enabled(False); torch.set_num_threads(8)
path=sys.argv[1]; tmax=float(sys.argv[2])
rows=[json.loads(l) for l in open(path,encoding='utf-8') if l.strip()]
live=np.stack([_unity_quats_to_smpl(r["unityLocalQuaternions"]) for r in rows if r.get("type")=="prediction"])
cur=replay_live(path, simulate=True)
print("rebases",cur.rebases, "sync frames", cur.synced.sum())
off=LyingCatContact().offsets
def m(local):
    g,p=fk(local,off)
    chest=g[9]@np.array([0,0,1.]); 
    d=[np.linalg.norm(p[j]-p[15]) for j in (20,21)]
    back=[(p[j]-p[15])@chest for j in (20,21)]   # <0: behind (under) the head
    return d+back
t=cur.times
for name,loc in (("live",live),("current",cur.final_local)):
    M=np.array([m(x) for x in loc])
    for lo,hi in ((0,tmax),):
        k=(t>=lo)&(t<hi)&(t>cur.rebases[0][0] if cur.rebases else (t>=lo)&(t<hi))
        print(f"{name:8s} {lo}-{hi}s n={k.sum()} wrist-head dist L/R median {np.median(M[k,0])*100:.0f}/{np.median(M[k,1])*100:.0f} cm  p90 {np.percentile(M[k,0],90)*100:.0f}/{np.percentile(M[k,1],90)*100:.0f}  wrist ahead of head (chest dir) L/R {np.median(M[k,2])*100:+.0f}/{np.median(M[k,3])*100:+.0f} cm")
np.save(path.replace('.jsonl','_cur_final.npy'), cur.final_local) if False else None
lo,hi=float(sys.argv[3]),float(sys.argv[4])
for name,loc in (("live",live),("current",cur.final_local),("current-raw-net",cur.raw_local)):
    M=np.array([m(x) for x in loc]); k=(t>=lo)&(t<hi)
    G=[fk(x,off)[0] for x in loc[k]]
    # forearm direction (elbow->wrist) in pelvis frame
    fl=np.array([g[0].T@g[18]@np.array([1,0,0.]) for g in G]); fr=np.array([g[0].T@g[19]@np.array([-1,0,0.]) for g in G])
    print(f"{name:15s} {lo}-{hi}s wrist-head L/R med {np.median(M[k,0])*100:.0f}/{np.median(M[k,1])*100:.0f} cm (p10 {np.percentile(M[k,0],10)*100:.0f}/{np.percentile(M[k,1],10)*100:.0f})  forearm dir in pelvis frame L {np.round(np.median(fl,0),2)} R {np.round(np.median(fr,0),2)}")
