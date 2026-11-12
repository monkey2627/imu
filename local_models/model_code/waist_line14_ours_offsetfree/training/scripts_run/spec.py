"""Where in FREQUENCY does the joint-position error of the Joints module live?  A complementary filter that fuses the (accurate at high
frequency) relative acceleration with the network position can only fix error bands in which the acceleration is informative (roughly > 0.3-0.5 Hz);
a quasi-static / slowly varying error (wrong posture, swivel ambiguity) is invisible to it."""
import sys, numpy as np, torch
from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.eval_variant import Pred
from mobileposer.evaluate_standard import sequences, load_pnp_articulate, DATASETS
from mobileposer.evaluate_drift_anchorpose import prepare_imu
dev = torch.device('cuda:0'); body = ParametricModel(paths.smpl_file)
pred = Pred(sys.argv[1], sys.argv[2], dev); art = load_pnp_articulate()
FPS = 60
bands = [(0, 0.2), (0.2, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 30)]
names = {'elbow': [18, 19], 'wrist': [20, 21], 'knee': [4, 5], 'ankle': [7, 8], 'shoulder': [16, 17]}
for ds in DATASETS[:2]:
    acc = {k: np.zeros(len(bands)) for k in names}; tot = {k: 0.0 for k in names}; gtp = {k: np.zeros(len(bands)) for k in names}
    for name, a, R, pose in sequences(ds, art):
        imu = prepare_imu(a, R).to(dev).unsqueeze(0); T = imu.shape[1]
        if T < 256: continue
        with torch.no_grad():
            pj = pred.joints(imu, [T]).reshape(T, 24, 3).cpu()
        G, J = body.forward_kinematics(pose, torch.zeros(10), tran=None, calc_mesh=False); J = J.reshape(T, 24, 3)
        err = pj - J                                                     # root-relative position error
        f = np.fft.rfftfreq(T, 1 / FPS)
        for k, idx in names.items():
            E = np.fft.rfft(err[:, idx].numpy() - err[:, idx].numpy().mean(0), axis=0); P = (np.abs(E) ** 2).sum((1, 2))
            Gt = np.fft.rfft(J[:, idx].numpy() - J[:, idx].numpy().mean(0), axis=0); PG = (np.abs(Gt) ** 2).sum((1, 2))
            for b, (lo, hi) in enumerate(bands):
                m = (f >= lo) & (f < hi)
                acc[k][b] += P[m].sum() / T; gtp[k][b] += PG[m].sum() / T
            tot[k] += err[:, idx].pow(2).sum(-1).mean().item() * 0 + 1
    print(ds, '  fraction of the position-error energy per band (Hz):', bands)
    for k in names:
        print(f'  {k:9s} err ' + ' '.join(f'{x / acc[k].sum():5.2f}' for x in acc[k]) + '   | GT motion ' + ' '.join(f'{x / gtp[k].sum():5.2f}' for x in gtp[k]) + f'   | err/GT energy ratio per band ' + ' '.join(f'{a / max(g, 1e-9):5.2f}' for a, g in zip(acc[k], gtp[k])))
