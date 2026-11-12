"""Does a kinematic hinge constraint help the unsensed limb?  (no training)

wrists_shanks_waist senses the SHANKS and FOREARMS but not the thighs / upper arms.
Knee (and, roughly, elbow) is a one-axis hinge: R_thigh = R_shank * R_knee^T with R_knee a
rotation about ONE fixed local axis. The network predicts the thigh orientation freely (3 DoF).
Step 1: how hinge-like are GT knee/elbow local rotations (AMASS)?  principal axis + fraction of
        rotation-vector energy on that axis.
Step 2: on the 10 real captures' reference poses (IMU synthesized from the reference), take the
        network's predicted local knee/elbow rotation, project it onto the principal axis, rebuild
        the thigh / upper-arm global orientation from the predicted shank / forearm, and compare
        errors before/after.  Also report the GT's own hinge-violation floor (project GT, compare to GT).
Baseline (world-frame) and waist-frame-canonical predictors are both evaluated.

Run from SparseIMU/:  python -m mobileposer.diagnose_hinge_projection
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
from mobileposer.evaluate_canon_models import Predictors, group_err
from mobileposer.evaluate_drift_anchorpose import LAYOUT_NAME
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from mobileposer.orientation_fusion import _so3_log, _so3_exp
from drift_eval_common import load_long_sequences, angle_between_rotmats

# (child joint carrying the sensor, hinge joint whose local rot links parent->child, parent limb joint)
# knee: local rot of joint 4/5 (shank rel. thigh); thigh = joints 1/2.  elbow: joint 18/19 (forearm rel. upper arm); upper arm = 16/17.
HINGES = {'knee': ([4, 5], [1, 2]), 'elbow': ([18, 19], [16, 17])}


def principal_axis(rotvecs):
    X = rotvecs.reshape(-1, 3)
    X = X[X.norm(dim=-1) > 0.05]
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    ax = Vt[0]
    frac = float((S[0] ** 2) / (S ** 2).sum())
    return ax, frac


def project_hinge(R_local, axis):
    """project local rotations onto one-axis rotations about `axis` (angle = component of the rotvec on the axis)"""
    v = _so3_log(R_local)
    ang = (v * axis).sum(-1, keepdim=True)
    return _so3_exp(axis * ang)


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)

    # ---- step 1: hinge-likeness of GT (AMASS held-out + training-style sequences)
    seqs = load_long_sequences(min_frames=1800, max_seqs=30)
    allp = torch.cat([s['pose'].reshape(-1, 24, 3, 3) for s in seqs], 0)
    axes = {}
    print('=== step 1: GT hinge-likeness (30 held-out AMASS long sequences) ===')
    for name, (hj, _) in HINGES.items():
        for side, j in zip(('L', 'R'), hj):
            ax, frac = principal_axis(_so3_log(allp[:, j]))
            axes[(name, side)] = ax
            print(f'  {name} {side} (joint {j}): principal axis {[round(float(x), 2) for x in ax]}  energy on axis {100 * frac:.1f}%')

    # ---- step 2: projection on the 10 real captures' reference poses
    P = Predictors(dev)
    root = Path('results/ours_multi_finetune/captures')
    out = {}
    for cap in sorted(root.iterdir(), key=lambda p: int(p.name.split('_')[0])):
        t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        gt = t['pose'].reshape(-1, 24, 3, 3); T = len(gt)
        ori, _, acc = _synthesize(body, gt, torch.ones(10), torch.zeros(T, 3), LAYOUT_NAME, syn, 128)
        for mname, fn in (('baseline', P.baseline), ('canon', P.canon)):
            pred = fn(acc, ori)[:T]
            pg, _ = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
            pg = pg.reshape(-1, 24, 3, 3)
            gg, _ = body.forward_kinematics(gt, torch.ones(10), tran=None, calc_mesh=False)
            gg = gg.reshape(-1, 24, 3, 3)
            for name, (hj, lj) in HINGES.items():
                before, after, floor = [], [], []
                for side, h, l in zip(('L', 'R'), hj, lj):
                    ax = axes[(name, side)]
                    # predicted: parent limb orientation rebuilt from predicted child + projected hinge
                    R_hinge_pred = pred[:, h]
                    R_par_new = pg[:, h] @ project_hinge(R_hinge_pred, ax).transpose(-1, -2)
                    before.append(angle_between_rotmats(pg[:, l], gg[:, l]).mean().item())
                    after.append(angle_between_rotmats(R_par_new, gg[:, l]).mean().item())
                    # floor: project the GT's own hinge, rebuild from GT child
                    R_par_gt = gg[:, h] @ project_hinge(gt[:, h], ax).transpose(-1, -2)
                    floor.append(angle_between_rotmats(R_par_gt, gg[:, l]).mean().item())
                out.setdefault((mname, name), []).append((cap.name.split('__')[0], np.mean(before), np.mean(after), np.mean(floor)))

    print('\n=== step 2: parent-limb (thigh / upper arm) global orientation error, deg, mean over the 10 captures ===')
    print('model     hinge   before   after(hinge-projected)   GT hinge-violation floor')
    for (mname, name), rows in out.items():
        b, a, f = (np.mean([r[i] for r in rows]) for i in (1, 2, 3))
        print(f'{mname:9s} {name:6s} {b:7.1f}   {a:7.1f}                  {f:6.1f}')
    print('\nper exercise, baseline knee/thigh before -> after:')
    for ex, b, a, f in out[('baseline', 'knee')]:
        print(f'  {ex:14s} thigh {b:5.1f} -> {a:5.1f}   (floor {f:4.1f})')


if __name__ == '__main__':
    main()
