import json, sys, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from production.live_replay import replay_live, _unity_quats_to_smpl
from analysis.compare_video import fk
from production.lyingcat_contact import LyingCatContact
off=LyingCatContact().offsets
f=sys.argv[1]; lo,hi=float(sys.argv[2]),float(sys.argv[3])
rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
live=np.stack([_unity_quats_to_smpl(r["unityLocalQuaternions"]) for r in rows if r.get("type")=="prediction"])
cur=replay_live(f, simulate=True); t=cur.times; k=(t>=lo)&(t<hi)
for name,L in (("live",live),("current",cur.final_local)):
    M=[]
    for loc in L[k]:
        g,p=fk(loc,off)
        M.append((np.degrees(np.arcsin(g[9][1,1])), (p[15]-p[0])[1], np.linalg.norm(p[20]-p[15]), np.linalg.norm(p[21]-p[15])))
    M=np.array(M)
    print(f"{name:8s} {lo}-{hi}s spine3 elev p10/50/90 {np.percentile(M[:,0],[10,50,90]).round()}  head above pelvis p90 {np.percentile(M[:,1],90):.2f} m  wrist-head L/R med {np.median(M[:,2])*100:.0f}/{np.median(M[:,3])*100:.0f} p90 {np.percentile(M[:,2],90)*100:.0f}/{np.percentile(M[:,3],90)*100:.0f} cm")
