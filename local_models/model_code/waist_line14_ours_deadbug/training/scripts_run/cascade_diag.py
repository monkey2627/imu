"""Cascade exposure check: the Poser is TRAINED on noisy ground-truth joint positions (sigma 0.04 m) but TESTED on the Joints module's predictions.
Standard-protocol metrics (SIP/Ang/Pos/Mesh, offline, 30 fps) of the same Poser fed with (a) predicted joints, (b) GT + N(0,0.04) joints, (c) exact GT joints.
(c) is an oracle diagnostic only (not a method)."""
import sys, os, numpy as np, torch
from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.eval_variant import Pred
from mobileposer.evaluate_standard import sequences, load_pnp_articulate, StdEvaluator, DATASETS
from mobileposer.evaluate_drift_anchorpose import prepare_imu
dev = torch.device('cuda:0'); body = ParametricModel(paths.smpl_file); ev = StdEvaluator(dev)
pred = Pred(sys.argv[1], sys.argv[2], dev)
for ds in DATASETS[:2]:
    res = {k: [] for k in ('pred joints', 'GT+noise', 'GT joints')}
    for name, a, R, pose in sequences(ds, ev.art):
        imu = prepare_imu(a, R).to(dev).unsqueeze(0); T = imu.shape[1]
        with torch.no_grad():
            pj = pred.joints(imu, [T])
            G, J = body.forward_kinematics(pose, torch.zeros(10), tran=None, calc_mesh=False)
            Jgt = J.reshape(T, 72).to(dev).unsqueeze(0)
            for k, joints in (('pred joints', pj), ('GT+noise', Jgt + 0.04 * torch.randn_like(Jgt)), ('GT joints', Jgt)):
                out = pred.poser._reduced_global_to_full(pred.poser(torch.cat((joints, imu), -1), [T])).cpu()
                res[k].append(ev(out[:T], pose))
    print(ds, {k: np.round(torch.stack(v).mean(0).numpy(), 2).tolist() for k, v in res.items()}, flush=True)
