import sys, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from production.live_replay import replay_live
from analysis.compare_video import fk
from production.lyingcat_contact import LyingCatContact
off=LyingCatContact().offsets
f=sys.argv[1]; lo,hi=float(sys.argv[2]),float(sys.argv[3])
import production.curlup_contact as cc
if len(sys.argv)>4: cc.ELBOW_PRIOR_WEIGHT=float(sys.argv[4])
cur=replay_live(f, simulate=True); t=cur.times; k=np.where((t>=lo)&(t<hi))[0]
U={18:[],19:[]}; W=[]
for i in k:
    g,p=fk(cur.final_local[i],off)
    for e,s in ((18,16),(19,17)):
        u=g[9].T@g[s]@off[e]; U[e].append(u/np.linalg.norm(u))
    W.append((np.linalg.norm(p[20]-p[15]),np.linalg.norm(p[21]-p[15])))
W=np.array(W)*100; dt=np.diff(t[k])
for e,name in ((18,"L"),(19,"R")):
    u=np.array(U[e]); step=np.degrees(np.arccos(np.clip((u[1:]*u[:-1]).sum(1),-1,1)))/np.maximum(dt,1e-3)
    z=u[:,2]; y=u[:,1]
    print(f"{name} upper arm (chest frame): up(z) p10/p90 {np.percentile(z,10):+.2f}/{np.percentile(z,90):+.2f}  head(y) p10/p90 {np.percentile(y,10):+.2f}/{np.percentile(y,90):+.2f}  speed p50/p95 {np.percentile(step,50):.0f}/{np.percentile(step,95):.0f} deg/s")
print(f"wrist-head L/R med {np.median(W,0).round()} p90 {np.percentile(W,90,axis=0).round()} cm")
