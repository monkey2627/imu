import os, torch, sys
os.environ['MOBILEPOSER_ARCH']='tf'
from mobileposer.eval_variant import Pred, WinPred
from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_dip_real import load_dip
dev=torch.device('cuda:0'); body=ParametricModel(paths.smpl_file)
f=Pred('v3_tf_poser','v3_tf_joints',dev); w=WinPred(f,125,62)
s=load_dip(body)[0]; a,o=s['acc'][:600],s['ori'][:600]
p1=f(a,o); p2=w(a,o); p3=f(a[:125],o[:125])
print('full vs win diff', (p1-p2).abs().max().item(), 'full[:125] vs short', (p1[:125]-p3).abs().max().item(), 'std over time of full pred', p1.reshape(600,-1).std(0).mean().item())
x=torch.randn(1,200,132).to(dev); print('input sensitivity', (f.poser(x,[200])-f.poser(x*0,[200])).abs().mean().item())
print('train mode diff', end=' ')
f.poser.train(); print((f.poser(x,[200])-f.poser(x*0,[200])).abs().mean().item())
