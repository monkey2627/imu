import sys, numpy as np, torch
import production.curlup_contact as cc
from production.live_replay import replay_live
torch.set_grad_enabled(False); torch.set_num_threads(8)
log=[]
orig=cc.CurlUpContact._hand_behind_head
def dbg(side,g,p,shoulder,elbow,upper,offsets,supine):
    forearm=g[elbow]@offsets[elbow+2]
    te=p[15]+g[15]@cc.HAND_BEHIND_HEAD[side]-forearm; reach=te-p[shoulder]
    res=np.linalg.norm(reach)-np.linalg.norm(offsets[elbow])
    inward=(g[9].T@forearm)[0]/np.linalg.norm(forearm)*(-1 if side=="left" else 1)
    r=orig(side,g,p,shoulder,elbow,upper,offsets,supine); log.append((side,res,inward,supine,r[1])); return r
cc.CurlUpContact._hand_behind_head=staticmethod(dbg)
cur=replay_live(sys.argv[1],simulate=True)
L=np.array([(1 if s=="left" else 0,a,b,c,d) for s,a,b,c,d in log])
for s in (1,0):
    M=L[L[:,0]==s]; n=len(M); M=M[n//2:]  # later half ~ crunch phase
    print("left" if s else "right", "residual pct", np.round(np.percentile(M[:,1],[10,50,90]),3), "inward", np.round(np.percentile(M[:,2],[10,50,90]),2), "supine",np.round(np.median(M[:,3]),2),"weight", np.round(np.percentile(M[:,4],[10,50,90]),2))
