"""Which joint group's POSITION estimate limits the pose error?  Oracle diagnostic (not a method): the Poser of pnp5_base gets the
Joints module's prediction except for one joint group whose positions are replaced by ground truth (+0.04 m noise like in training)."""
import sys, numpy as np, torch
from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.eval_variant import Pred
from mobileposer.evaluate_standard import sequences, StdEvaluator, DATASETS
from mobileposer.evaluate_drift_anchorpose import prepare_imu
dev = torch.device('cuda:0'); body = ParametricModel(paths.smpl_file); ev = StdEvaluator(dev)
pred = Pred(sys.argv[1], sys.argv[2], dev)
GROUPS = {'none (all predicted)': [], 'legs (hips, knees, ankles, feet)': [1, 2, 4, 5, 7, 8, 10, 11], 'arms (collars, shoulders, elbows, wrists, hands)': [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
          'spine+head (spine1-3, neck, head)': [3, 6, 9, 12, 15], 'hips only': [1, 2], 'shoulders only': [16, 17], 'all (oracle)': list(range(24))}
for ds in DATASETS[:2]:
    res = {k: [] for k in GROUPS}
    for name, a, R, pose in sequences(ds, ev.art):
        imu = prepare_imu(a, R).to(dev).unsqueeze(0); T = imu.shape[1]
        with torch.no_grad():
            pj = pred.joints(imu, [T]).reshape(1, T, 24, 3)
            G, J = body.forward_kinematics(pose, torch.zeros(10), tran=None, calc_mesh=False)
            Jgt = J.reshape(1, T, 24, 3).to(dev); Jgt = Jgt + 0.04 * torch.randn_like(Jgt)
            for k, idx in GROUPS.items():
                m = pj.clone()
                if idx:
                    m[:, :, idx] = Jgt[:, :, idx]
                out = pred.poser._reduced_global_to_full(pred.poser(torch.cat((m.reshape(1, T, 72), imu), -1), [T])).cpu()
                res[k].append(ev(out[:T], pose))
    print(ds)
    for k, v in res.items():
        print(f'  {k:55s}', np.round(torch.stack(v).mean(0).numpy(), 2).tolist(), flush=True)
