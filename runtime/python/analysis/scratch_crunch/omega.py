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
O=np.asarray(O,float); T=np.asarray(T)
w=labels.index("waist")
# world angular increments (rotation vectors), smoothed over 6 frames
def wvec(k, step=6):
    Rt=R.from_matrix(O[:,k]); return (Rt[step:]*Rt[:-step].inv()).as_rotvec()
W=wvec(w)
spine=O[:-6,w]@np.array([0,1.,0]); head=np.stack([spine[:,0],0*spine[:,0],spine[:,2]],1); head/=np.linalg.norm(head,axis=1,keepdims=True)
left=-np.cross(np.array([0,1.,0]),head)  # supine: body-left; check sign below with waist x axis
wx=O[:-6,w]@np.array([1.,0,0]); print("waist +x vs computed 'left' dot:", np.mean(np.sum(wx*left,1)))
def yawd(v): return np.degrees(np.arctan2(v[...,0],v[...,2]))
def pca(V):
    m=np.linalg.norm(V,axis=1)>np.radians(2)
    u,s,vt=np.linalg.svd(V[m],full_matrices=False); return vt[0], s/np.sum(s), m.sum()
ax,sv,n=pca(W); print("waist omega main axis", np.round(ax,2), "yaw rel head", round(float(np.mean(((yawd(ax)-yawd(head))+180)%360-180)),1), "energy",np.round(sv,2), n)
Lw=W@left[0]
for l in ("lw","rw"):
    k=labels.index(l); V=wvec(k)
    ax,sv,n=pca(V)
    relyaw=((yawd(ax)-yawd(head[0]))+180)%360-180
    # orient forearm axis to correlate positively with waist flexion component
    s=np.sign(np.sum((V@ax)*Lw)); ax=ax*s
    relyaw=((yawd(ax)-yawd(left[0]))+180)%360-180
    corr=np.corrcoef(V@ax, Lw)[0,1]
    print(l,"omega axis",np.round(ax,2),"el",round(float(np.degrees(np.arcsin(ax[1]))),1),"yaw rel body-left(+x waist)",round(float(relyaw),1),"energy",np.round(sv,2),"corr w/ waist flex",round(corr,2),
          "gain", round(float(np.sum((V@ax)*Lw)/np.sum(Lw*Lw)),2))
