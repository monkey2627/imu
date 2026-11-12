"""Is the failure in lying-type postures a WORLD-ORIENTATION problem rather than an
articulation problem?  Take held-out upright walking sequences (articulation the model
handles), rigidly rotate the WHOLE body (and its translation) by an angle about a horizontal
axis, re-synthesize the 5-IMU signals, and compare predictions with the rotated ground truth.
If error explodes with rotation angle while articulation is unchanged, the model's dependence
on absolute world orientation is a method-addressable failure mechanism.
Controls: 0 deg, yaw (vertical-axis) rotation. Errors: UNALIGNED global orientation per group (deg).
Run from SparseIMU/:  python -m mobileposer.diagnose_orientation_sensitivity
"""
import math
import sys
from pathlib import Path
import numpy as np
import torch
_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent)); sys.path.insert(0, str(_DIR.parent.parent))
from mobileposer.config import paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_drift_anchorpose import prepare_imu, LAYOUT_NAME
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from drift_eval_common import load_long_sequences, angle_between_rotmats

GROUPS = {'pelvis[0]': [0], 'SENSED[4,5,18,19]': [4, 5, 18, 19], 'thigh[1,2]': [1, 2], 'spine2/3[6,9]': [6, 9],
          'neck/head[12-15]': [12, 13, 14, 15], 'upperarm[16,17]': [16, 17]}


def rot(axis, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    if axis == 'x': return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
    if axis == 'y': return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
body = ParametricModel(paths.smpl_file)
model = load_model('checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth').to(dev).eval()
syn = synthesis_metadata('surface', DEFAULT_CONFIG)
seqs = load_long_sequences(min_frames=1800, max_seqs=12)
conds = [('0 (as is)', None), ('yaw(y) 90', rot('y', 90)), ('pitch(x) 45', rot('x', 45)), ('pitch(x) 90', rot('x', 90)),
         ('roll(z) 90', rot('z', 90)), ('pitch(x) 180 (upside down)', rot('x', 180))]
res = {c: [] for c, _ in conds}
for s in seqs:
    pose = s['pose'].clone()[:900].reshape(-1, 24, 3, 3); tran = s['tran'].clone()[:900]
    for name, R in conds:
        p, tr = pose.clone(), tran.clone()
        if R is not None:
            p[:, 0] = R @ p[:, 0]
            tr = (R @ tr.T).T
        with torch.no_grad():
            ori, _, acc = _synthesize(body, p, torch.ones(10), tr, LAYOUT_NAME, syn, 128)
            imu = prepare_imu(acc, ori).to(dev)
            model.reset()
            pred, _, _, _ = model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
            pred = (pred.squeeze(0) if pred.dim() == 4 else pred).cpu().reshape(-1, 24, 3, 3)
            T = min(len(pred), len(p))
            gp, _ = body.forward_kinematics(pred[:T], torch.ones(10), tran=None, calc_mesh=False)
            gq, _ = body.forward_kinematics(p[:T], torch.ones(10), tran=None, calc_mesh=False)
            e = angle_between_rotmats(gp.reshape(T, 24, 3, 3), gq.reshape(T, 24, 3, 3)).mean(0).numpy()
        res[name].append([float(e[js].mean()) for js in GROUPS.values()])
print('condition'.ljust(30) + '  '.join(n[:17].rjust(17) for n in GROUPS))
for name, _ in conds:
    print(name.ljust(30) + '  '.join(f'{x:17.1f}' for x in np.mean(res[name], axis=0)))
print(f'\n({len(seqs)} held-out upright sequences, first 900 frames; UNALIGNED global orientation error, deg)')
