"""Check of evaluate_ours_captures' sensed-bone numbers: root-aligned error is measured
relative to the PELVIS, so a wrong predicted pelvis orientation contaminates every joint.
Separate: (1) pelvis (joint 0) global orientation error; (2) UNALIGNED global orientation
error of the sensed bones (4,5,18,19) -- directly comparable to what the IMUs measure;
(3) root-aligned error of the same bones. IMU is synthesized from the reference pose.
Run from SparseIMU/:  python -m mobileposer.diagnose_root_vs_sensed
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

SENSED_BONES = [4, 5, 18, 19]
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
body = ParametricModel(paths.smpl_file)
model = load_model('checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth').to(dev).eval()
syn = synthesis_metadata('surface', DEFAULT_CONFIG)
print('exercise'.ljust(26) + 'pelvis(0)  sensed-bones UNALIGNED  sensed-bones ROOT-ALIGNED')
rows = []
for cap in sorted(Path('results/ours_multi_finetune/captures').iterdir(), key=lambda p: int(p.name.split('_')[0])):
    t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
    gt = t['pose'].reshape(-1, 24, 3, 3); T = len(gt)
    with torch.no_grad():
        ori, _, acc = _synthesize(body, gt, torch.ones(10), torch.zeros(T, 3), 'wrists_shanks_waist', syn, 128)
        imu = prepare_imu(acc, ori).to(dev)
        model.reset()
        pred, _, _, _ = model.forward_offline(imu.unsqueeze(0), [T])
        pred = (pred.squeeze(0) if pred.dim() == 4 else pred).cpu().reshape(-1, 24, 3, 3)[:T]
        gp, _ = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
        gq, _ = body.forward_kinematics(gt, torch.ones(10), tran=None, calc_mesh=False)
        gp, gq = gp.reshape(T, 24, 3, 3), gq.reshape(T, 24, 3, 3)
        pelvis = angle_between_rotmats(gp[:, 0], gq[:, 0]).mean().item()
        unal = angle_between_rotmats(gp[:, SENSED_BONES], gq[:, SENSED_BONES]).mean().item()
        pa, ga = pred.clone(), gt.clone(); pa[:, 0] = torch.eye(3); ga[:, 0] = torch.eye(3)
        hp, _ = body.forward_kinematics(pa, torch.ones(10), tran=None, calc_mesh=False)
        hq, _ = body.forward_kinematics(ga, torch.ones(10), tran=None, calc_mesh=False)
        ral = angle_between_rotmats(hp.reshape(T, 24, 3, 3)[:, SENSED_BONES], hq.reshape(T, 24, 3, 3)[:, SENSED_BONES]).mean().item()
    rows.append((pelvis, unal, ral))
    print(cap.name.split('__')[0].ljust(26) + f'{pelvis:7.1f}     {unal:10.1f}               {ral:10.1f}')
r = np.array(rows); print('mean'.ljust(26) + f'{r[:,0].mean():7.1f}     {r[:,1].mean():10.1f}               {r[:,2].mean():10.1f}')
