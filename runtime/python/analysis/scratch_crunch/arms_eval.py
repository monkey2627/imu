import sys, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from production.live_replay import replay_live
from analysis.compare_video import fk
from production.lyingcat_contact import LyingCatContact
off=LyingCatContact().offsets
f=sys.argv[1]; phases=[(float(a),float(b),n) for a,b,n in (x.split(":") for x in sys.argv[2:])]
cur=replay_live(f, simulate=True); t=cur.times
for lo,hi,name in phases:
    k=np.where((t>=lo)&(t<hi))[0]
    for e,s,side,sg in ((18,16,"L",1),(19,17,"R",-1)):
        U=[];W=[]
        for i in k:
            g,p=fk(cur.final_local[i],off)
            u=g[9].T@g[s]@off[e]; u/=np.linalg.norm(u); u[0]*=sg; U.append(u); W.append(np.linalg.norm(p[e+2]-p[15]))
        U=np.array(U); sp=np.degrees(np.arccos(np.clip((U[1:]*U[:-1]).sum(1),-1,1)))/np.maximum(np.diff(t[k]),1e-3)
        print(f"{name:10s} {side} upper arm (out,head,up) p10 {np.percentile(U,10,axis=0).round(2)} p50 {np.median(U,0).round(2)} p90 {np.percentile(U,90,axis=0).round(2)} | speed p50/p95 {np.percentile(sp,50):.0f}/{np.percentile(sp,95):.0f} deg/s | wrist-head med {np.median(W)*100:.0f} cm")
