import json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from production.calibration import bone_orientation
from production.input_recording import _sample_for
from production.pose_calibration import BONE_AXES
f=sys.argv[1]; t0=float(sys.argv[2]); t1=float(sys.argv[3])
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
se=el(O[:,w]@np.array([0,1.,0]))
# segment curls: local maxima of spineEl > 20 with preceding minimum < 13
from scipy.signal import find_peaks
pk,_=find_peaks(se,height=20,distance=40,prominence=8)
for p in pk:
    lo=max(0,p-120); m=lo+np.argmin(se[lo:p])
    if se[m]>14: continue
    rw_=R.from_matrix(O[p,w])*R.from_matrix(O[m,w]).inv(); av=rw_.as_rotvec()
    left=O[m,w]@np.array([1.,0,0])
    s=f"{T[m]:5.1f}->{T[p]:5.1f} waist {np.degrees(np.linalg.norm(av)):4.0f}deg axis.left {av@left/np.linalg.norm(av):5.2f} |"
    for l in ("lw","rw"):
        k=labels.index(l); rv=(R.from_matrix(O[p,k])*R.from_matrix(O[m,k]).inv()).as_rotvec()
        ang=np.degrees(np.linalg.norm(rv)); ax=rv/np.linalg.norm(rv)
        D=(yaw(ax)-yaw(av)+180)%360-180
        b0=O[m,k]@BONE_AXES[l]
        s+=f" {l} {ang:4.0f}deg axEl {el(ax):4.0f} yaw(axis vs waist axis) {D:5.0f} el {el(b0):4.0f}->{el(O[p,k]@BONE_AXES[l]):4.0f} rawry {(yaw(b0)-yaw(O[m,w]@np.array([0,1.,0]))+180)%360-180:5.0f} |"
    print(s)
