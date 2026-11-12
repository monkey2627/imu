"""Corrected diagnostic: UNALIGNED global orientation error per joint group on the 10 real
captures' reference poses (IMU synthesized from the reference, wrists_shanks_waist), with the
pelvis (joint 0) error reported separately. Root-aligned error is contaminated by pelvis error.
Modes: normal (Poser fed Joints-module prediction) vs oracle (Poser fed GT joints + 0.04 noise).
Run from SparseIMU/:  python -m mobileposer.diagnose_unaligned_by_group
"""
import sys
from pathlib import Path
import numpy as np
import torch
_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent)); sys.path.insert(0, str(_DIR.parent.parent))
from mobileposer.config import paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_drift_anchorpose import prepare_imu
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from drift_eval_common import angle_between_rotmats

GROUPS = {'pelvis[0]': [0], 'SENSED shank/forearm[4,5,18,19]': [4, 5, 18, 19], 'waist spine1[3]': [3],
          'UNS thigh[1,2]': [1, 2], 'UNS spine2/3[6,9]': [6, 9], 'UNS neck/head/clav[12-15]': [12, 13, 14, 15],
          'UNS upperarm[16,17]': [16, 17]}
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
body = ParametricModel(paths.smpl_file)
model = load_model('checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth').to(dev).eval()
syn = synthesis_metadata('surface', DEFAULT_CONFIG)
names = list(GROUPS)


def glob_err(pred_local, gt_local, T):
    gp, _ = body.forward_kinematics(pred_local, torch.ones(10), tran=None, calc_mesh=False)
    gq, _ = body.forward_kinematics(gt_local, torch.ones(10), tran=None, calc_mesh=False)
    return angle_between_rotmats(gp.reshape(T, 24, 3, 3), gq.reshape(T, 24, 3, 3)).mean(0).numpy()


out = {'normal': [], 'oracle': []}
for cap in sorted(Path('results/ours_multi_finetune/captures').iterdir(), key=lambda p: int(p.name.split('_')[0])):
    t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
    gt = t['pose'].reshape(-1, 24, 3, 3); T = len(gt)
    with torch.no_grad():
        ori, _, acc = _synthesize(body, gt, torch.ones(10), torch.zeros(T, 3), 'wrists_shanks_waist', syn, 128)
        imu = prepare_imu(acc, ori).to(dev).unsqueeze(0)
        _, gj = body.forward_kinematics(gt, torch.ones(10), tran=None, calc_mesh=False)
        gj = gj.reshape(1, T, 72).to(dev)
        model.reset()
        pj = model.joints(imu, [T])
        pa = model._reduced_global_to_full(model.pose(torch.cat((pj, imu), -1), [T])).cpu().reshape(-1, 24, 3, 3)
        pb = model._reduced_global_to_full(model.pose(torch.cat((gj + 0.04 * torch.randn_like(gj), imu), -1), [T])).cpu().reshape(-1, 24, 3, 3)
    for mode, p in (('normal', pa), ('oracle', pb)):
        e = glob_err(p, gt, T)
        out[mode].append((cap.name.split('__')[0], [float(e[js].mean()) for js in GROUPS.values()]))
for mode in ('normal', 'oracle'):
    print(f'\n=== {mode}: UNALIGNED global orientation error (deg) ===')
    print('exercise'.ljust(24) + '  '.join(n[:16].rjust(16) for n in names))
    for ex, v in out[mode]:
        print(ex[:23].ljust(24) + '  '.join(f'{x:16.1f}' for x in v))
    m = np.mean([v for _, v in out[mode]], axis=0)
    print('mean'.ljust(24) + '  '.join(f'{x:16.1f}' for x in m))
