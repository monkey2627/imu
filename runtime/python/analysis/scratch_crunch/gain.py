import json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks
from production.calibration import bone_orientation
from production.input_recording import _sample_for
from production.pose_calibration import BONE_AXES
f=sys.argv[1]; t0=float(sys.argv[2]); t1=float(sys.argv[3])
rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
labels=rows[0]["labels"]; T=[];O=[]
for r in rows:
    if r.get("type")=="tick" and t0<=r["t"]<=t1:
        s={l:_sample_for(r["samples"][l],l) for l in labels}
        O.append(bone_orientation(np.asarray([s[l].quat_xyzw for l in labels],np.float32),labels).numpy()); T.append(r["t"])
O=np.asarray(O,float); T=np.asarray(T); w=labels.index("waist")
el=lambda v: np.degrees(np.arcsin(np.clip(v[...,1],-1,1)))
se=el(O[:,w]@np.array([0,1.,0]))
pk,_=find_peaks(se,height=se.min()+5,distance=40,prominence=4)
ratios=[]
for p in pk:
    lo=max(0,p-120); m=lo+np.argmin(se[lo:p])
    left=O[m,w]@np.array([1.,0,0])
    rows_=[]
    for l in ("lw","rw"):
        k=labels.index(l); rv=(R.from_matrix(O[p,k])*R.from_matrix(O[m,k]).inv()).as_rotvec()
        rows_.append(rv@left)   # signed rotation about body-left (flexion +)
    fw=np.degrees(np.mean(rows_)); ww=se[p]-se[m]
    typ="open(co-rotate)" if fw>10 else "elbows-in" if fw<-10 else "?"
    if fw>10: ratios.append(fw/ww)
    print(f"{T[m]:5.1f}->{T[p]:5.1f} waist +{ww:4.1f}  forearm flex {fw:+5.0f}  {typ}")
print("open-elbow ratio forearm/waist: median %.2f  range %s" % (np.median(ratios), np.round([min(ratios),max(ratios)],2)) if ratios else "none")
