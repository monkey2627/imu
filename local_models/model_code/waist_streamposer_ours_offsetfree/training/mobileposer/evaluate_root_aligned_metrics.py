"""Root-aligned LOCAL and GLOBAL (SIP-style) angular error on the 30 held-out
long AMASS sequences, optionally with a PoserHier swapped in for the baseline
Poser (Joints / Velocity / FootContact stay the baseline checkpoint's).

"Global, root-aligned" follows the DIP/TransPose/PIP/PNP convention: set the
root rotation to identity in BOTH prediction and GT, then forward-kinematics,
so a joint's global rotation is expressed relative to the pelvis frame. Under
this convention a hip's global error == its local error (parent = root).

Run from SparseIMU/:
    python -m mobileposer.evaluate_root_aligned_metrics
    python -m mobileposer.evaluate_root_aligned_metrics --hier-ckpt checkpoints/poser_hier_wsw/1/poser/<file>.ckpt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

_DIR = Path(__file__).resolve().parent
_BASE = _DIR.parent
_CODE = _BASE.parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_CODE))

from mobileposer.config import paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_drift_anchorpose import to_training_basis, prepare_imu, LAYOUT_NAME
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from drift_eval_common import load_long_sequences, angle_between_rotmats

GROUPS = {
    'root[0] (local=global rot)': [0],
    'hips[1,2]': [1, 2],
    'knees[4,5]': [4, 5],
    'lumbar5[1,2,3,6,9]': [1, 2, 3, 6, 9],
    'SIP[1,2,16,17]': [1, 2, 16, 17],
    'all15': [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19],
}


def per_joint_errors(bodymodel, pose_pred, gt, shape):
    """returns (local[T,24], global_root_aligned[T,24]) in degrees."""
    local = angle_between_rotmats(pose_pred, gt)
    pp, gg = pose_pred.clone(), gt.clone()
    pp[:, 0] = torch.eye(3)
    gg[:, 0] = torch.eye(3)
    gp, _ = bodymodel.forward_kinematics(pp, shape, tran=None, calc_mesh=False)
    gt_g, _ = bodymodel.forward_kinematics(gg, shape, tran=None, calc_mesh=False)
    return local, angle_between_rotmats(gp.reshape(-1, 24, 3, 3), gt_g.reshape(-1, 24, 3, 3))


def run(model, bodymodel, synthesis, sequences, device):
    out = []
    for seq in sequences:
        gt_pose, gt_tran = to_training_basis(seq['pose'], seq['tran'])
        shape = torch.ones(10)
        with torch.no_grad():
            if 'acc' in seq:   # pre-synthesized (DIP) -- use as stored
                acc, ori = seq['acc'], seq['ori']
            else:
                ori, joint, acc = _synthesize(bodymodel, gt_pose, shape, gt_tran, LAYOUT_NAME, synthesis, 128)
            imu = prepare_imu(acc, ori).to(device)
            model.reset()
            pose_pred, _, _, _ = model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
            pose_pred = pose_pred.squeeze(0).cpu() if pose_pred.dim() == 4 else pose_pred.cpu()
            T = min(len(gt_pose), len(pose_pred))
            l, g = per_joint_errors(bodymodel, pose_pred[:T].reshape(T, 24, 3, 3), gt_pose[:T].reshape(T, 24, 3, 3), shape)
        out.append((l.numpy(), g.numpy()))
    return out


def summarize(name, res):
    print(f'\n--- {name} ---')
    table = {}
    for gname, js in GROUPS.items():
        loc = np.array([r[0][:, js].mean() for r in res])
        glo = np.array([r[1][:, js].mean() for r in res])
        table[gname] = (loc, glo)
        print(f'{gname:22s} local={loc.mean():6.3f}  global(root-aligned)={glo.mean():6.3f}')
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min_frames', type=int, default=1800)
    ap.add_argument('--max_seqs', type=int, default=30)
    ap.add_argument('--checkpoint', default=str(paths.root_dir / 'checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth'))
    ap.add_argument('--hier-ckpt', nargs='*', default=None)
    ap.add_argument('--dip-file', default=None, help='pre-synthesized DIP-GT layout data, e.g. data/no_head_5imu_surface_dip_test/wrists_shanks_waist/dip_test.pt')
    ap.add_argument('--min_dip_frames', type=int, default=300)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)
    model.eval()
    bodymodel = ParametricModel(paths.smpl_file)
    synthesis = synthesis_metadata('surface', DEFAULT_CONFIG)
    if args.dip_file:
        data = torch.load(args.dip_file, map_location='cpu')
        sequences = [{'acc': a, 'ori': o, 'pose': p, 'tran': t}
                     for a, o, p, t in zip(data['acc'], data['ori'], data['pose'], data['tran'])]
        sequences = [s for s in sequences if len(s['pose']) >= args.min_dip_frames]
        print(f'Loaded {len(sequences)} DIP-GT sequences (IMU synthesized for this layout)', flush=True)
    else:
        sequences = load_long_sequences(min_frames=args.min_frames, max_seqs=args.max_seqs)
        print(f'Loaded {len(sequences)} long sequences', flush=True)

    base = summarize('baseline Poser', run(model, bodymodel, synthesis, sequences, device))

    from mobileposer.models.poser_hier import PoserHier
    for ck in (args.hier_ckpt or []):
        model.pose = PoserHier.load_from_checkpoint(ck, map_location=device).to(device).eval()
        hier = summarize(f'PoserHier {ck}', run(model, bodymodel, synthesis, sequences, device))
        print('\n--- paired comparison (per-sequence, hier vs baseline; + = hier better) ---')
        for gname in GROUPS:
            for k, label in ((0, 'local'), (1, 'global')):
                b, h = base[gname][k], hier[gname][k]
                try:
                    p = wilcoxon(b, h).pvalue
                except ValueError:
                    p = float('nan')
                print(f'{gname:22s} {label:6s} {b.mean():6.3f} -> {h.mean():6.3f}  '
                      f'({100 * (1 - h.mean() / b.mean()):+.2f}%)  improved {(h < b).sum()}/{len(b)}  p={p:.2e}')


if __name__ == '__main__':
    main()
