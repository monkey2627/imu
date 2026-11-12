import sys, torch, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'..')
from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_dip_real import load_dip
from mobileposer.evaluate_causal_holds import load
from mobileposer.diagnose_latch_heuristic import apply_latch, thigh_errors, THIGH
from drift_eval_common import angle_between_rotmats
dev=torch.device('cuda:0'); body=ParametricModel(paths.smpl_file)
src=load('checkpoints/v3_causal125_poser','checkpoints/v3_causal125_joints',True,dev)
def oracle_gate(G, open_s, thr):
    flex=angle_between_rotmats(G[:,1],G[:,4])
    ev=torch.zeros(len(flex),dtype=torch.bool); ev[30:]=(flex[30:]-flex[:-30]).abs()>thr   # GT event: >thr deg change within 1 s
    n=int(open_s*30); mv=torch.zeros_like(ev)
    idx=torch.nonzero(ev).flatten()
    for i in idx: mv[i:i+n]=True
    mv[:15]=True
    return mv
res={}
for s in load_dip(body):
    T=len(s['pose']); G,J=body.forward_kinematics(s['pose'],torch.ones(10),tran=None,calc_mesh=False); G,J=G.reshape(T,24,3,3),J.reshape(T,24,3)
    rest=[(G[:,ti].transpose(-1,-2)@(J[:,ti+3]-J[:,ti]).unsqueeze(-1)).squeeze(-1).mean(0) for ti,_ in THIGH]
    Gp,_=body.forward_kinematics(src(s['acc'],s['ori'])[:T],torch.ones(10),tran=None,calc_mesh=False); Gp=Gp.reshape(T,24,3,3)
    for name,(os_,thr) in {'none':None,'oracle open 3s':(3,25),'oracle open 5s':(5,25),'oracle open 8s':(8,25),'oracle open 5s thr15':(5,15)}.items() if False else []: pass
    cfg={'none':None,'oracle open 2s':(2,25),'oracle open 4s':(4,25),'oracle open 8s':(8,25)}
    for name,c in cfg.items():
        es=[];ds=[]
        for w,(ti,si) in enumerate(THIGH):
            th=Gp[:,ti] if c is None else apply_latch(Gp,s['ori'],oracle_gate(G,*c),ti,si)
            e,d=thigh_errors(body,G,th,J,rest,w); es.append(e); ds.append(d)
        res.setdefault(name,[]).append((s['name'],T,torch.stack(es).mean().item(),torch.stack(ds).mean().item()))
print('oracle (GT-event) latch on causal125, DIP real; thigh orientation/direction error deg')
for name,rows in res.items():
    w=np.array([r[1] for r in rows],float)
    allm=np.average([[r[2],r[3]] for r in rows],axis=0,weights=w)
    sit=[r for r in rows if r[0]=='s_09/05.pkl'][0]
    s10=[r for r in rows if r[0]=='s_10/02.pkl'][0]
    print(f'  {name:16s} all {allm[0]:5.1f}/{allm[1]:5.1f}   sit s_09/05 {sit[2]:5.1f}/{sit[3]:5.1f}   s_10/02 {s10[2]:5.1f}/{s10[3]:5.1f}')
