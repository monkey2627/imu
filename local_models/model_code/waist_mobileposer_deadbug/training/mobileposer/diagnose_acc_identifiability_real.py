"""Same identifiability test as diagnose_acc_identifiability.py, but on the REAL measurements of the
10 own captures: measured relative acceleration (left shank sensor - waist sensor, world linear acc)
vs the second derivative of the hypothesised relative position v(theta,t), with the sensed bone
orientations taken from the REAL sensor orientations (oracle-calibrated, or as given).
  v(theta,t) = R_w h' + R_s Rknee(theta)^T t_k + R_s b_s     (pelvis orientation approximated by the waist sensor's)
Constants h', t_k, b_s come from the SMPL-neutral reference geometry. theta_true = reference (pseudo-label) knee flexion.
Left leg, windows of 16 frames, constant theta per window.
Run from SparseIMU/:  python -m mobileposer.diagnose_acc_identifiability_real
"""
import sys
from pathlib import Path
import numpy as np
import torch
_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent)); sys.path.insert(0, str(_DIR.parent.parent))
from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.no_head_layouts import LAYOUTS, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG, surface_kinematics
from mobileposer.orientation_fusion import _so3_log, _so3_exp
from mobileposer.diagnose_acc_identifiability import KNEE_AXIS, THETAS, FPS, W

body = ParametricModel(paths.smpl_file)
syn = synthesis_metadata('surface', DEFAULT_CONFIG)
layout = LAYOUTS['wrists_shanks_waist']
Rk = _so3_exp(THETAS.view(-1, 1) * KNEE_AXIS.view(1, 3))                      # [K,3,3]
print('REAL IMU: knee flexion from measured relative acceleration (left leg), window', W)
print('exercise        cond          windows  excited(>0.3)  median|err| excited   within 20deg   network knee-flex err (ref)')
net_ref = {'1_卧位猫': 44, '2_坐位': 82, '3_站位': 2, '4_臀桥': 19, '5_侧桥': 57, '6_卷腹': 26, '7_死虫': 45, '8_鸟狗': 52, '9_屈膝': 61, '10_站立位伸展': 10}
for cap in sorted(Path('results/ours_multi_finetune/captures').iterdir(), key=lambda p: int(p.name.split('_')[0])):
    ex = cap.name.split('__')[0]
    t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
    pose = t['pose'].reshape(-1, 24, 3, 3)
    d = torch.load(cap / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
    acc = d['acc'][0]; ori_raw = d['ori'][0].clone(); ori_cal = ori_raw.clone()
    T = min(len(pose), len(acc), len(t['global_pose']))
    pose, acc, ori_raw, ori_cal = pose[:T], acc[:T], ori_raw[:T], ori_cal[:T]
    for k, j in enumerate(d['layout_joints']):
        M = (ori_raw[:, k].transpose(-1, -2) @ t['global_pose'][:T, j]).mean(0)
        U, S, Vt = torch.linalg.svd(M)
        if torch.det(U @ Vt) < 0: U[:, -1] *= -1
        ori_cal[:, k] = ori_raw[:, k] @ (U @ Vt)
    G, J = body.forward_kinematics(pose.reshape(-1, 216), torch.ones(10), tran=None, calc_mesh=False)
    G, J = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
    _, _, pts = surface_kinematics(body, pose, torch.ones(10), torch.zeros(T, 3), layout['labels'], syn['attachments'], 128)
    p_s, p_w = pts[:, 2], pts[:, 4]
    hprime = (G[:, 3].transpose(-1, -2) @ (J[:, 1] - p_w).unsqueeze(-1)).squeeze(-1).mean(0)
    t_k = (G[:, 1].transpose(-1, -2) @ (J[:, 4] - J[:, 1]).unsqueeze(-1)).squeeze(-1).mean(0)
    b_s = (G[:, 4].transpose(-1, -2) @ (p_s - J[:, 4]).unsqueeze(-1)).squeeze(-1).mean(0)
    th_true = (_so3_log(pose[:, 4]) * KNEE_AXIS).sum(-1)
    for cname, ori in (('oracle-calib', ori_cal), ('as-given', ori_raw)):
        Rw, Rs = ori[:, 4], ori[:, 2]
        v = (Rw @ hprime).unsqueeze(0) + (Rs.unsqueeze(0) @ Rk.transpose(-1, -2).unsqueeze(1) @ t_k.view(1, 1, 3, 1)).squeeze(-1) + (Rs @ b_s).unsqueeze(0)
        a_hyp = (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) * FPS * FPS
        a_meas = (acc[:, 2] - acc[:, 4])[1:-1]                                  # measured relative acceleration (ls - waist)
        errs, exc = [], []
        for s in range(0, T - 2 - W, W // 2):
            sl = slice(s, s + W)
            r = ((a_hyp[:, sl] - a_meas[sl].unsqueeze(0)) ** 2).sum(-1).mean(1)
            kk = int(torch.argmin(r))
            errs.append(abs(float(torch.rad2deg(THETAS[kk] - th_true[s + 1:s + 1 + W].mean()))))
            exc.append(float(a_meas[sl].norm(dim=-1).mean()))
        errs, exc = np.array(errs), np.array(exc); hi = exc > 0.3
        print(f'{ex:14s}  {cname:12s}  {len(errs):6d}   {100 * hi.mean():8.0f}%      {np.median(errs[hi]) if hi.any() else float("nan"):9.1f}        '
              f'{100 * (errs[hi] < 20).mean() if hi.any() else float("nan"):6.0f}%        {net_ref[ex]}')
