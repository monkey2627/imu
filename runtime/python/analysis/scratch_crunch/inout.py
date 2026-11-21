import json, sys
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
from analysis.compare_video import fk
from production.live_replay import replay_live
from production.lyingcat_contact import LyingCatContact
from production.calibration import bone_orientation
from production.input_recording import _sample_for
from production.start_pose_fit import TEMPLATES, find_hold, fit
from production.rotmat import UNITY_SMPL_BASIS
torch.set_grad_enabled(False); torch.set_num_threads(8)
path=sys.argv[1]; th0,th1=float(sys.argv[2]),float(sys.argv[3]); lo,hi=float(sys.argv[4]),float(sys.argv[5])
rows=[json.loads(l) for l in open(path,encoding='utf-8') if l.strip()]
labels=rows[0]["labels"]
T=[];O=[]
for r in rows:
    if r.get("type")!="tick": continue
    s={l:_sample_for(r["samples"][l],l) for l in labels}
    O.append(bone_orientation(np.asarray([s[l].quat_xyzw for l in labels],np.float32),labels).numpy()); T.append(r["t"])
O=np.asarray(O,float); T=np.asarray(T)
tpl=TEMPLATES[("crunch","supine")]
k=(T>=th0)&(T<=th1); holds=find_hold(O[k],labels,tpl)
out,rej=fit(tpl,holds,holds["waist"]); print("fit rej",rej)
C=O.copy()
for l,(H,M,*_) in out.items():
    i=labels.index(l); C[:,i]=H@O[:,i]@M
cur=replay_live(path, simulate=True)
off=LyingCatContact().offsets
w=labels.index("waist")
for t0 in np.arange(lo,hi,0.5):
    j=np.argmin(abs(cur.times-t0)); i=np.argmin(abs(T-cur.times[j]))
    s=f"{cur.times[j]:6.2f}"
    for name,loc in (("net",cur.raw_local),("final",cur.final_local)):
        g,p=fk(loc[j],off)
        s+=f" | {name} wristHead {np.linalg.norm(p[20]-p[15])*100:3.0f}/{np.linalg.norm(p[21]-p[15])*100:3.0f}"
        if name=="net":
            # forearm relative to pelvis: output vs input (deg)
            e=[]
            for l,jj in (("lw",18),("rw",19)):
                a=g[0].T@g[jj]; b=C[i,w].T@C[i,labels.index(l)]
                e.append(np.degrees(R.from_matrix(a.T@b).magnitude()))
            s+=f" fore(out vs in) {e[0]:3.0f}/{e[1]:3.0f}°"
    sp=C[i,w]@np.array([0,1.,0]); s+=f" | waist spineEl {np.degrees(np.arcsin(sp[1])):4.0f}"
    print(s)
