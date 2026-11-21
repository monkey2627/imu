import json, sys
import numpy as np
from production.calibration import bone_orientation
from production.input_recording import _sample_for
from production.pose_calibration import BONE_AXES
f=sys.argv[1]; hold=(float(sys.argv[2]),float(sys.argv[3])); lo,hi=float(sys.argv[4]),float(sys.argv[5])
rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
labels=rows[0]["labels"]; T=[];O=[]
for r in rows:
    if r.get("type")=="tick":
        s={l:_sample_for(r["samples"][l],l) for l in labels}
        O.append(bone_orientation(np.asarray([s[l].quat_xyzw for l in labels],np.float32),labels).numpy()); T.append(r["t"])
O=np.asarray(O,float); T=np.asarray(T)
el=lambda v: np.degrees(np.arcsin(np.clip(v[...,1],-1,1)))
w=el(O[:,4]@np.array([0,1.,0])); fe=0.5*(el(O[:,0]@BONE_AXES["lw"])+el(O[:,1]@BONE_AXES["rw"]))
h=(T>=hold[0])&(T<hold[1]); w0=np.median(w[h]); f0=np.median(fe[h])
k=(T>=lo)&(T<hi)
dw=w[k]-w0; df=fe[k]-f0
print(f"{f.split('/')[-1]}: hold waist {w0:.1f} forearm {f0:.1f} | corr(waist change, forearm change) {np.corrcoef(dw,df)[0,1]:+.2f}")
for thr in (15,):
    up=df>thr; dn=df<-10
    print(f"  forearm >hold+{thr}: {up.mean()*100:.0f}% frames, waist change there med {np.median(dw[up]):+.1f} | forearm <hold-10: {dn.mean()*100:.0f}%, waist change med {np.median(dw[dn]):+.1f} | else {np.median(dw[~up&~dn]):+.1f}")
