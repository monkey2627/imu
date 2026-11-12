"""Is the knee flexion IDENTIFIABLE from the relative acceleration of the waist and shank sensors?
(no network, no training; pure kinematics on the reference poses)

Chain: pelvis -> hip -> thigh (NO sensor) -> knee -> shank (sensor).  With the orientations of the waist
(spine1) and shank known, the thigh orientation is R_thigh(theta) = R_shank * Rknee(theta)^T for a knee
flexion theta (one-axis hinge). The shank sensor position relative to the waist sensor is then
    v(theta,t) = [J_hip + R_thigh(theta) t_k + R_shank b_s] - p_waist
so the measured relative acceleration a_meas = d2/dt2 (p_shank - p_waist) must equal d2/dt2 v(theta,t).
Per window (constant theta assumed over W frames) we grid-search theta and compare with the true mean theta.
Constants (hip offset in pelvis frame, knee offset in thigh frame, sensor offsets in bone frames) are
estimated from the same reference and are CONSTANT by construction. Noise-free and with Gaussian accelerometer
noise (sigma per sensor, relative noise = sqrt(2) sigma). Left leg. Windows with too little relative acceleration
are reported separately (no information there).

Run from SparseIMU/:  python -m mobileposer.diagnose_acc_identifiability
"""
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.no_head_layouts import LAYOUTS, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG, surface_kinematics
from mobileposer.orientation_fusion import _so3_log, _so3_exp
from drift_eval_common import load_long_sequences

KNEE_AXIS = torch.tensor([0.99, -0.01, -0.15])
KNEE_AXIS = KNEE_AXIS / KNEE_AXIS.norm()
FPS = 30.0
W = 16
THETAS = torch.deg2rad(torch.arange(-10, 151, 4).float())


def second_diff(x):
    return (x[2:] - 2 * x[1:-1] + x[:-2]) * FPS * FPS


def analyse(body, syn, pose, name, sigma_list=(0.0, 0.2, 0.5)):
    T = len(pose)
    layout = LAYOUTS['wrists_shanks_waist']
    G, J = body.forward_kinematics(pose.reshape(-1, 216), torch.ones(10), tran=None, calc_mesh=False)
    G, J = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
    ori, _, points = surface_kinematics(body, pose, torch.ones(10), torch.zeros(T, 3), layout['labels'], syn['attachments'], 128)
    p_s = points[:, 2]          # left shank sensor point (label order lw, rw, ls, rs, waist)
    p_w = points[:, 4]          # waist sensor point
    # constants
    h = (G[:, 0].transpose(-1, -2) @ (J[:, 1] - J[:, 0]).unsqueeze(-1)).squeeze(-1).mean(0)               # hip in pelvis frame
    t_k = (G[:, 1].transpose(-1, -2) @ (J[:, 4] - J[:, 1]).unsqueeze(-1)).squeeze(-1).mean(0)             # knee in thigh frame
    b_s = (G[:, 4].transpose(-1, -2) @ (p_s - J[:, 4]).unsqueeze(-1)).squeeze(-1).mean(0)                 # shank sensor in shank frame
    theta_true = (_so3_log(pose[:, 4]) * KNEE_AXIS).sum(-1)                                                # knee flexion (rad)
    a_true = second_diff(p_s - p_w)                                                                         # relative acceleration [T-2,3]
    # hypotheses
    Rk = _so3_exp(THETAS.view(-1, 1) * KNEE_AXIS.view(1, 3))                                               # [K,3,3]
    Gth = G[:, 4].unsqueeze(0) @ Rk.transpose(-1, -2).unsqueeze(1)                                         # [K,T,3,3] thigh orientation hypotheses
    Jhip = J[:, 0] + (G[:, 0] @ h.unsqueeze(-1)).squeeze(-1)                                               # [T,3]
    v = Jhip.unsqueeze(0) + (Gth @ t_k.view(1, 1, 3, 1)).squeeze(-1) + (G[:, 4] @ b_s.view(3, 1)).squeeze(-1).unsqueeze(0) - p_w.unsqueeze(0)
    a_hyp = (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) * FPS * FPS                                            # [K,T-2,3]
    rng = np.random.RandomState(0)
    res = {}
    for sigma in sigma_list:
        noise = torch.from_numpy(rng.randn(*a_true.shape) * sigma * np.sqrt(2)).float()
        a_meas = a_true + noise
        errs, excite = [], []
        for s in range(0, T - 2 - W, W // 2):
            sl = slice(s, s + W)
            exc = a_true[sl].norm(dim=-1).mean().item()
            r = ((a_hyp[:, sl] - a_meas[sl].unsqueeze(0)) ** 2).sum(-1).mean(1)                              # [K]
            k = int(torch.argmin(r))
            errs.append(abs(float(torch.rad2deg(THETAS[k] - theta_true[s + 1:s + 1 + W].mean()))))
            excite.append(exc)
        res[sigma] = (np.array(errs), np.array(excite))
    return res


def report(name, res):
    for sigma, (e, x) in res.items():
        hi = x > 0.3        # windows with relative acceleration > 0.3 m/s^2 (rms-ish) carry signal
        print(f'{name:22s} noise {sigma:3.1f}  windows {len(e):4d}  with excitation(>0.3m/s2) {100 * hi.mean():4.0f}%   '
              f'median |theta err| all {np.median(e):5.1f}  excited {np.median(e[hi]) if hi.any() else float("nan"):5.1f}  '
              f'within 15deg: excited {100 * (e[hi] < 15).mean() if hi.any() else float("nan"):4.0f}%')


def main():
    body = ParametricModel(paths.smpl_file)
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)
    print('knee flexion identifiability from relative waist-shank acceleration (left leg), window', W, 'frames\n')
    for cap in sorted(Path('results/ours_multi_finetune/captures').iterdir(), key=lambda p: int(p.name.split('_')[0])):
        t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        pose = t['pose'].reshape(-1, 24, 3, 3)[:2400]
        r = analyse(body, syn, pose, cap.name.split('__')[0])
        report(cap.name.split('__')[0], {k: v for k, v in r.items() if k in (0.0, 0.2)})
    print('\nreference: upright walking (AMASS BioMotionLab_NTroje long sequences)')
    seqs = load_long_sequences(min_frames=1800, max_seqs=3)
    for i, s in enumerate(seqs):
        r = analyse(body, syn, s['pose'].reshape(-1, 24, 3, 3)[:1500], f'walk{i}')
        report(f'walk{i}', {k: v for k, v in r.items() if k in (0.0, 0.2)})


if __name__ == '__main__':
    main()
