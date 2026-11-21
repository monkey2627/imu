import json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from production.calibration import bone_orientation
from production.input_recording import _sample_for
from production.pose_calibration import BONE_AXES
f=sys.argv[1]; t0,t1=float(sys.argv[2]),float(sys.argv[3])
rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
labels=rows[0]["labels"]; O=[]
for r in rows:
    if r.get("type")=="tick" and t0<=r["t"]<=t1:
        s={l:_sample_for(r["samples"][l],l) for l in labels}
        O.append(bone_orientation(np.asarray([s[l].quat_xyzw for l in labels],np.float32),labels).numpy())
O=np.asarray(O,float); H={l:R.from_matrix(O[:,i]).mean().as_matrix() for i,l in enumerate(labels)}
feet=sum(H[l]@BONE_AXES[l] for l in ("ls","rs")); feet[1]=0; feet/=np.linalg.norm(feet)
head=-feet; up=np.array([0,1.,0]); left=np.cross(head,up)  # supine body-left: head x up? check sign below
# SMPL pelvis columns: x=body-left, y=spine(head), z=chest
for sgn in (1,-1):
    E=np.stack([sgn*left,head,up],1)
    if np.linalg.det(E)>0: break
M=H["waist"].T@E
rv=R.from_matrix(M).as_rotvec(); print(f.split('/')[-1], "waist mount error", round(np.degrees(np.linalg.norm(rv)),1), "deg about sensor axis", np.round(rv/np.linalg.norm(rv),2))
print(" euler xyz", np.round(R.from_matrix(M).as_euler('xyz',degrees=True),1))
from production.pose_calibration import _yaw
best=min(((np.degrees(R.from_matrix((_yaw(-a)@H["waist"]@_yaw(a)).T@E).magnitude()),np.degrees(a)) for a in np.radians(np.arange(-180,180,1))))
print(f" mounting-yaw hypothesis: waist mount yaw {best[1]:+.0f} deg leaves {best[0]:.1f} deg residual")
