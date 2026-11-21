import json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from production.calibration import bone_orientation
from production.input_recording import _sample_for
from production.pose_calibration import BONE_AXES
from production.start_pose_fit import head_yaw, TEMPLATES, fit, elevation_error
f=sys.argv[1]; t0=float(sys.argv[2]); t1=float(sys.argv[3])
rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
labels=rows[0]["labels"]
O=[]
for r in rows:
    if r.get("type")!="tick" or not (t0<=r["t"]<=t1): continue
    s={l:_sample_for(r["samples"][l],l) for l in labels}
    O.append(bone_orientation(np.asarray([s[l].quat_xyzw for l in labels],np.float32),labels).numpy())
O=np.asarray(O,float)
holds={l:R.from_matrix(O[:,i]).mean().as_matrix() for i,l in enumerate(labels)}
F=head_yaw(holds["waist"])
np.set_printoptions(precision=2,suppress=True)
for l in ("lw","rw"):
    B=F.T@holds[l]
    print(l,"body-frame cols x,y,z:\n",B, "\n bone",B@BONE_AXES[l], "palm(-y)",-B[:,1])
tpl=TEMPLATES[("crunch","supine")]
out,rej=fit(tpl,holds,holds["waist"])
for l,v in out.items(): print(l, f"yaw {v[2]:+.0f} swing {v[3]:.0f} twist {v[4]:.0f} elerr {elevation_error(l,tpl[l],holds[l]):.0f}", "REJ" if l in rej else "")
