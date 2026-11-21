import json, sys
import numpy as np
from production.calibration import bone_orientation
from production.input_recording import _sample_for
from production.pose_calibration import BONE_AXES
f=sys.argv[1]; t0=float(sys.argv[2]); t1=float(sys.argv[3]); dt=float(sys.argv[4]) if len(sys.argv)>4 else 0
rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
labels=rows[0]["labels"]
T=[];O=[]
for r in rows:
    if r.get("type")!="tick" or not (t0<=r["t"]<=t1): continue
    s={l:_sample_for(r["samples"][l],l) for l in labels}
    q=np.asarray([s[l].quat_xyzw for l in labels],np.float32)
    O.append(bone_orientation(q,labels).numpy()); T.append(r["t"])
O=np.asarray(O,float); T=np.asarray(T); w=labels.index("waist")
el=lambda v: np.degrees(np.arcsin(np.clip(v[...,1],-1,1)))
yaw=lambda v: np.degrees(np.arctan2(v[...,0],v[...,2]))
sp=O[:,w]@np.array([0,1.,0]); se=el(sp)
d=lambda x: x[6:]-x[:-6]
print(f"{t0}-{t1}s waist spineEl range {se.min():.0f}..{se.max():.0f}")
for l in ("lw","rw"):
    k=labels.index(l); b=O[:,k]@BONE_AXES[l]; fe=el(b); ry=(yaw(b)-yaw(sp)+180)%360-180
    c=np.corrcoef(d(fe),d(se))[0,1]; g=np.polyfit(d(se),d(fe),1)[0]
    pn=O[:,k]@np.array([0,-1.,0])  # palm normal (T-pose palm -y)
    print(f" {l}: el {fe.min():.0f}..{fe.max():.0f}  raw relyaw median {np.median(ry):.0f}  palm-normal up median {np.median(pn[:,1]):.2f}  corr(d el, d spineEl) {c:.2f} slope {g:.2f}")
if dt:
    for t in np.arange(t0,t1,dt):
        i=np.searchsorted(T,t); 
        if i>=len(T): break
        s=f"{T[i]:6.2f} spineEl {se[i]:5.1f}"
        for l in ("lw","rw"):
            k=labels.index(l); b=O[i,k]@BONE_AXES[l]
            s+=f" | {l} el {el(b):5.1f} ry {((yaw(b)-yaw(sp[i])+180)%360-180):5.0f} palmUp {(O[i,k]@np.array([0,-1.,0]))[1]:5.2f}"
        print(s)
