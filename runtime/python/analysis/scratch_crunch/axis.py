import json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from production.calibration import bone_orientation
from production.input_recording import _sample_for
from production.pose_calibration import BONE_AXES
f=sys.argv[1]; tmax=float(sys.argv[2])
rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
labels=rows[0]["labels"]; print(labels)
T=[];O=[]
for r in rows:
    if r.get("type")!="tick" or r["t"]>tmax: continue
    s={l:_sample_for(r["samples"][l],l) for l in labels}
    q=np.asarray([s[l].quat_xyzw for l in labels],np.float32)
    O.append(bone_orientation(q,labels).numpy()); T.append(r["t"])
O=np.asarray(O,float); T=np.asarray(T)
w=labels.index("waist")
def el(v): return np.degrees(np.arcsin(np.clip(v[...,1],-1,1)))
def yaw(v): return np.degrees(np.arctan2(v[...,0],v[...,2]))
spine=O[:,w]@np.array([0,1,0.]); chest=O[:,w]@np.array([0,0,1.])
for t0 in np.arange(0,tmax,1.0):
    i=np.searchsorted(T,t0)
    if i>=len(T): break
    out=f"{t0:5.0f}s waist spineEl {el(spine[i]):5.0f} yaw {yaw(spine[i]):5.0f} chestUp {chest[i,1]:5.2f} |"
    for l in ("lw","rw","ls","rs"):
        k=labels.index(l); b=O[i,k]@BONE_AXES[l]
        out+=f" {l} el {el(b):4.0f} relyaw {((yaw(b)-yaw(spine[i])+180)%360-180):5.0f}"
    print(out)
