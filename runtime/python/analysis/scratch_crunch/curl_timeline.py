import sys, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from production.live_replay import replay_live
from analysis.compare_video import fk
from production.lyingcat_contact import LyingCatContact
off=LyingCatContact().offsets
f=sys.argv[1]; lo,hi,step=float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
cur=replay_live(f, simulate=True); t=cur.times; last=-9
for i in range(len(t)):
    if t[i]<lo or t[i]>hi or t[i]-last<step: continue
    last=t[i]; g,p=fk(cur.final_local[i],off)
    print(f"{t[i]:5.1f} spine3 elev {np.degrees(np.arcsin(g[9][1,1])):4.0f}  head above pelvis {(p[15]-p[0])[1]*100:3.0f} cm  wrist-head {np.linalg.norm(p[20]-p[15])*100:3.0f}/{np.linalg.norm(p[21]-p[15])*100:3.0f}")
