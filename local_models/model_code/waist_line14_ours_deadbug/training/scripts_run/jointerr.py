import sys, torch, numpy as np
from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.eval_variant import Pred
from mobileposer.evaluate_standard import sequences, load_pnp_articulate, online_predict
from mobileposer.body_ctx import body_context
from drift_eval_common import angle_between_rotmats
dev = torch.device('cuda:0'); body = ParametricModel(paths.smpl_file); art = load_pnp_articulate()
pred = Pred('pnp5_base_poser', 'pnp5_base_joints', dev); ctx = body_context(torch.zeros(10))
NAMES = ['pelvis','L hip','R hip','spine1','L knee','R knee','spine2','L ankle','R ankle','spine3','L foot','R foot','neck','L collar','R collar','head','L shoulder','R shoulder','L elbow','R elbow']
for ds in ['dipimu', 'totalcapture_dipcalib']:
    errs, w = [], []
    for name, a, R, pose in sequences(ds, art):
        p = pred(a, R, ctx)[:len(pose)]
        Gp, _ = body.forward_kinematics(p, torch.ones(10), tran=None, calc_mesh=False)
        G, _ = body.forward_kinematics(pose, torch.ones(10), tran=None, calc_mesh=False)
        e = angle_between_rotmats(Gp.reshape(-1, 24, 3, 3), G.reshape(-1, 24, 3, 3)).mean(0).numpy()
        errs.append(e); w.append(len(pose))
    e = np.average(errs, axis=0, weights=w)
    print(ds, 'global angle error per joint (deg):')
    print('  ' + '  '.join(f'{NAMES[i]} {e[i]:.1f}' for i in range(20)))
    print(f'  SIP parts: hips {np.mean(e[[1,2]]):.1f}  shoulders {np.mean(e[[16,17]]):.1f}  -> SIP~{np.mean(e[[1,2,16,17]]):.1f}')
