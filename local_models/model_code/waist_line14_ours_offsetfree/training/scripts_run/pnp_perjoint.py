import qpsolvers
_orig_qp = qpsolvers.solve_qp
qpsolvers.solve_qp = lambda *a, **k: _orig_qp(*a, **dict(k, solver="cvxopt"))
import sys, types
try:
    import quaternion
except ImportError:
    sys.modules["quaternion"] = types.ModuleType("quaternion")
"""Run the released PNP network on the standard test sets and report standard metrics + per-joint global angle errors
(to see where its advantage over our model comes from).  Run from the PNP directory in the 'pnp' conda env."""
import sys, json, numpy as np, torch, tqdm
import articulate as art
from net import PNP
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = PNP().eval().to(dev)
ev = art.FullMotionEvaluator('models/SMPL_male.pkl', joint_mask=torch.tensor([1, 2, 16, 17]), device=dev)
body = art.ParametricModel('models/SMPL_male.pkl', device=dev)
ign = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
g = torch.tensor([0, -9.8, 0])
NAMES = ['pelvis','L hip','R hip','spine1','L knee','R knee','spine2','L ankle','R ankle','spine3','L foot','R foot','neck','L collar','R collar','head','L shoulder','R shoulder','L elbow','R elbow']
out = {}
for name in sys.argv[1:]:
    data = torch.load(f'data/test_datasets/{name}.pt')
    std, perj, w = [], [], []
    for i in range(len(data['pose'])):
        RMB = data['RIM'][i].transpose(1, 2).matmul(data['RIS'][i]).matmul(data['RSB'][i]).to(dev)
        aM = (data['RIM'][i].transpose(1, 2).matmul(data['RIS'][i]).matmul(data['aS'][i].unsqueeze(-1)).squeeze(-1) + g).to(dev)
        wM = data['RIM'][i].transpose(1, 2).matmul(data['RIS'][i]).matmul(data['wS'][i].unsqueeze(-1)).squeeze(-1).to(dev)
        pose = art.math.axis_angle_to_rotation_matrix(data['pose'][i]).view(-1, 24, 3, 3)
        net.rnn_initialize(pose[0]); net.pose_prediction = torch.zeros_like(pose); net.tran_prediction = torch.zeros_like(data['tran'][i])
        for t in range(pose.shape[0]):
            net.pose_prediction[t], net.tran_prediction[t] = net.forward_frame(aM[t], wM[t], RMB[t])
        p, q = net.pose_prediction.clone(), pose.clone().to(p.device if False else dev)
        pp, qq = p.clone().view(-1, 24, 3, 3).to(dev), q.clone().view(-1, 24, 3, 3).to(dev)
        pp[:, ign] = torch.eye(3, device=dev); qq[:, ign] = torch.eye(3, device=dev)
        e = ev(pose_p=pp, pose_t=qq)
        std.append(torch.stack([e[9, 0], e[3, 0], e[0, 0] * 100, e[1, 0] * 100]).cpu())
        Gp = body.forward_kinematics_R(body.inverse_kinematics_R(p.view(-1, 24, 3, 3)) if False else p.view(-1, 24, 3, 3)) if False else None
        # global rotations: predictions are local rotations -> FK
        Gp = body.forward_kinematics(p.view(-1, 24, 3, 3).to(dev), calc_mesh=False)[0].view(-1, 24, 3, 3)
        Gq = body.forward_kinematics(q.view(-1, 24, 3, 3).to(dev), calc_mesh=False)[0].view(-1, 24, 3, 3)
        ang = art.math.radian_to_degree(art.math.angle_between(Gp, Gq).view(-1, 24)) if hasattr(art.math, 'angle_between') else None
        perj.append(ang.mean(0).cpu().numpy()); w.append(len(pose))
        print(name, data['name'][i], np.round(std[-1].numpy(), 2), flush=True)
    s = torch.stack(std).mean(0).tolist(); pj = np.average(perj, axis=0, weights=w)
    out[name] = {'std': s, 'perjoint': {NAMES[k]: float(pj[k]) for k in range(20)}}
    print(name, 'SIP/Ang/Pos/Mesh', np.round(s, 2))
    print('  per joint:', ' '.join(f'{NAMES[k]} {pj[k]:.1f}' for k in range(20)))
    print('  hips %.1f shoulders %.1f' % (np.mean(pj[[1, 2]]), np.mean(pj[[16, 17]])))
json.dump(out, open("pnp_perjoint_" + "_".join(sys.argv[1:]) + ".json", "w"))
