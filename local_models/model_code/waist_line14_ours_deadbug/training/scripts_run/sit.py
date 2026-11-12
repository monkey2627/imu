import sys, torch, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'..')
from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_dip_real import load_dip
from mobileposer.evaluate_v2_models import ModulePredictor
from mobileposer.evaluate_causal_holds import load
from mobileposer.diagnose_dip_direction_vs_twist import vec_angle
from drift_eval_common import angle_between_rotmats
dev=torch.device('cuda:0'); body=ParametricModel(paths.smpl_file)
base=ModulePredictor('checkpoints/v3_base_poser','checkpoints/v3_base_joints',False,dev)
causal=load('checkpoints/v3_causal125_poser','checkpoints/v3_causal125_joints',True,dev); c1k=load('checkpoints/v3_causal1000_poser','checkpoints/v3_causal1000_joints',True,dev)
for s in load_dip(body):
    if s['name'] not in ('s_09/05.pkl','s_10/02.pkl'): continue
    T=len(s['pose']); G,J=body.forward_kinematics(s['pose'],torch.ones(10),tran=None,calc_mesh=False); G,J=G.reshape(T,24,3,3),J.reshape(T,24,3)
    flex=angle_between_rotmats(G[:,1],G[:,4]); up=torch.rad2deg(torch.acos(G[:,3,1,1].clamp(-1,1)))
    out={'base':base(s['acc'],s['ori'])[:T],'causal':causal(s['acc'],s['ori'])[:T],'c1k':c1k(s['acc'],s['ori'])[:T]}
    print(s['name'],'T',T)
    for k,p in out.items():
        Gp,Jp=body.forward_kinematics(p,torch.ones(10),tran=None,calc_mesh=False); Jp=Jp.reshape(T,24,3)
        d=vec_angle(J[:,4]-J[:,1],Jp[:,4]-Jp[:,1])
        if k=='base': dd={}
        dd[k]=d
    for a in range(0,T,150):
        b=min(T,a+150)
        print(f'  t={a/30:5.0f}-{b/30:5.0f}s flex(thigh-shank) {flex[a:b].mean():6.1f}  torso-tilt {up[a:b].mean():5.1f}  thigh-dir err base {dd["base"][a:b].mean():5.1f} causal125 {dd["causal"][a:b].mean():5.1f} causal1000 {dd["c1k"][a:b].mean():5.1f}')
