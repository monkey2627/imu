"""Jointly fine-tune the three AnchorPose modules (CameraFusionGRU,
OrientationFusionGRU, SwingFusionGRU) on top of the frozen backbone, starting
from their independently-trained checkpoints.

The modules read/write disjoint outputs, so with separate losses joint training
is mathematically identical to continued independent training. The only
genuine coupling is the `hold_global` re-expression of the root's children:
hip local = R_root_fused^T R_root_old (swing-corrected hip), so the hip loss
back-propagates into the orientation module. Modes:
  --mode control : each module trained on its OWN loss only (hip loss measured
                   BEFORE the root re-expression) == continued independent training
  --mode joint   : hip/knee loss measured AFTER hold_global (couples ori<->swing)
  --mode joint_mpjpe : joint + a world-frame joint-position loss through FK of the
                   fully assembled (fused pose, fused tran) output
Same dataset-level split and short-sequence data as the original module trainers.

Run from SparseIMU/ (env: mobileposer):
    python -m mobileposer.train_anchorpose_joint --mode joint --output runtime_joint_ft_joint
"""
import argparse
import json
import random
from pathlib import Path

import torch

from mobileposer.config import model_config, paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_no_head_5imu import load_sequences, prepare_imu
from mobileposer.train_camera_fusion import split_subsets
from mobileposer.camera_fusion import CameraFusionGRU
from mobileposer.orientation_fusion import OrientationFusionGRU
from mobileposer.joints_swing_fusion import (
    SwingFusionGRU, HIP_KNEE_JOINTS, build_rest_dir, compute_swing_twist_components,
)
from mobileposer.synthetic_slam import synthesize_slam_position, synthesize_slam_orientation
from mobileposer.train_joints_swing_fusion import geodesic_loss, angle_err_deg


@torch.no_grad()
def cache_one(base, bodymodel, rest_dir, seq, device):
    imu = prepare_imu(seq['acc'], seq['ori']).to(device)
    base.reset()
    pose, joints, tran, _ = base.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
    pose = (pose.squeeze(0) if pose.dim() == 4 else pose).cpu().reshape(-1, 24, 3, 3)
    joints = (joints.squeeze(0) if joints.dim() == 3 else joints).cpu().view(-1, 24, 3)
    tran = (tran.squeeze(0) if tran.dim() == 3 else tran).cpu()
    gp, gt = seq['pose'].float().reshape(-1, 24, 3, 3), seq['tran'].float()
    T = min(len(gp), len(pose), len(joints), len(tran), len(gt))
    pose, joints, tran, gp, gt = pose[:T], joints[:T], tran[:T], gp[:T], gt[:T]
    R_sp, R_st, R_tp = compute_swing_twist_components(bodymodel, pose, joints, rest_dir)
    return {'pose': pose, 'tran': tran - tran[:1], 'gp': gp, 'gt': gt - gt[:1],
            'sp': R_sp, 'st': R_st, 'tp': R_tp}


def forward_all(mods, c, device, rms_deg, tc_pos=0.3, seed=None):
    cam, ori, swing = mods
    gt_abs = c['gt']
    slam_pos = synthesize_slam_position(gt_abs, fps=30.0, target_rms_m=0.66, time_constant_s=tc_pos, seed=seed)
    slam_ori = synthesize_slam_orientation(c['gp'][:, 0], fps=30.0, target_rms_deg=rms_deg, seed=seed)
    b = lambda x: x.unsqueeze(0).to(device)
    tran_f, _, _ = cam(b(c['tran']), b(slam_pos))
    root_f, _, _ = ori(b(c['pose'][:, 0]), b(slam_ori))
    hk, _, _ = swing(b(c['sp']), b(c['st']), b(c['tp']))
    return tran_f.squeeze(0), root_f.squeeze(0), hk.squeeze(0)   # [T,3], [T,3,3], [T,4,3,3]


def losses(mods, c, device, mode, bodymodel, rms_deg, seed=None):
    tran_f, root_f, hk = forward_all(mods, c, device, rms_deg, seed=seed)
    d = lambda x: x.to(device)
    gt_pose, root_old = d(c['gp']), d(c['pose'][:, 0])
    l_tran = torch.nn.functional.mse_loss(tran_f, d(c['gt']))
    l_root = geodesic_loss(root_f, gt_pose[:, 0])
    hk_eval = hk
    delta = root_f.transpose(-1, -2) @ root_old                    # R_new^T R_old
    if mode != 'control':   # hold_global coupling: hips are root children (SMPL 1,2)
        hk_eval = hk.clone()
        hk_eval[:, 0] = delta @ hk[:, 0]
        hk_eval[:, 1] = delta @ hk[:, 1]
    l_hk = geodesic_loss(hk_eval, gt_pose[:, HIP_KNEE_JOINTS])
    total = l_tran + l_root + l_hk
    l_pos = torch.zeros(())
    if mode == 'joint_mpjpe':
        pose = d(c['pose']).clone()
        pose[:, HIP_KNEE_JOINTS] = hk_eval
        for ch in (3,):
            pose[:, ch] = delta @ pose[:, ch]
        pose[:, 0] = root_f
        _, jp = bodymodel.forward_kinematics(pose, torch.ones(10), tran=tran_f, calc_mesh=False)
        _, jg = bodymodel.forward_kinematics(gt_pose, torch.ones(10), tran=d(c['gt']), calc_mesh=False)
        l_pos = (jp - jg).norm(dim=-1).mean()
        total = total + l_pos
    return total, {'tran': l_tran.item(), 'root': l_root.item(), 'hk': l_hk.item(), 'pos': float(l_pos)}


def load_mods(args, device):
    def blob(p):
        return torch.load(p, map_location=device, weights_only=False)
    b = blob(args.cam_ckpt)
    cam = CameraFusionGRU(hidden=int(b.get('manifest', {}).get('hidden', 32))).to(device)
    cam.load_state_dict(b['cam_state_dict'] if 'cam_state_dict' in b else b['state_dict'])
    b = blob(args.ori_ckpt)
    ori = OrientationFusionGRU(hidden=int(b.get('manifest', {}).get('hidden', 32))).to(device)
    ori.load_state_dict(b['state_dict'])
    b = blob(args.swing_ckpt)
    swing = SwingFusionGRU(hidden=int(b.get('manifest', {}).get('hidden', 32))).to(device)
    swing.load_state_dict(b['state_dict'])
    return cam, ori, swing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['control', 'joint', 'joint_mpjpe'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--checkpoint', default='checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth')
    ap.add_argument('--data-root', default='data/no_head_5imu_surface_processed/wrists_shanks_waist')
    ap.add_argument('--cam-ckpt', default='runtime_camera_fusion_recal/last.pt')
    ap.add_argument('--ori-ckpt', default='runtime_orientation_fusion_dr/last.pt')
    ap.add_argument('--swing-ckpt', default='runtime_swing_fusion_v1/last.pt')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache-file', type=Path, default=None)
    ap.add_argument('--max-seq-per-subset', type=int, default=None)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_config.device = device
    bodymodel = ParametricModel(paths.smpl_file)
    data_root = Path(args.data_root)

    if args.cache_file is not None and args.cache_file.exists():
        blob = torch.load(args.cache_file, map_location='cpu', weights_only=False)
        cached, holdout_subsets = blob['cached'], blob['holdout_subsets']
    else:
        base = load_model(args.checkpoint).to(device).eval()
        for p in base.parameters():
            p.requires_grad_(False)
        rest_dir = build_rest_dir(bodymodel)
        split, holdout_subsets = split_subsets(data_root)
        cached = {'train': [], 'holdout': []}
        counts = {}
        for seq in load_sequences(data_root, max_seq=None):
            subset = seq['source'].split('[')[0]
            fold = split.get(subset)
            if fold is None:
                continue
            if args.max_seq_per_subset is not None and counts.get((subset, fold), 0) >= args.max_seq_per_subset:
                continue
            counts[(subset, fold)] = counts.get((subset, fold), 0) + 1
            c = cache_one(base, bodymodel, rest_dir, seq, device)
            c['source'] = seq['source']
            cached[fold].append(c)
        if args.cache_file is not None:
            torch.save({'cached': cached, 'holdout_subsets': holdout_subsets}, args.cache_file)
        del base
    print(json.dumps({'train': len(cached['train']), 'holdout': len(cached['holdout'])}), flush=True)

    mods = load_mods(args, device)
    params = [p for m in mods for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)

    def run_holdout():
        for m in mods:
            m.eval()
        acc = {'tran': [], 'root_deg': [], 'hk_deg': [], 'b_tran': [], 'b_root_deg': [], 'b_hk_deg': []}
        with torch.no_grad():
            for i, c in enumerate(cached['holdout']):
                tran_f, root_f, hk = forward_all(mods, c, device, rms_deg=3.0, seed=10_000 + i)
                gp = c['gp'].to(device)
                delta = root_f.transpose(-1, -2) @ c['pose'][:, 0].to(device)
                hk_eval = hk.clone()
                hk_eval[:, 0] = delta @ hk[:, 0]
                hk_eval[:, 1] = delta @ hk[:, 1]
                acc['tran'].append((tran_f - c['gt'].to(device)).norm(dim=-1).mean().item())
                acc['b_tran'].append((c['tran'] - c['gt']).norm(dim=-1).mean().item())
                acc['root_deg'].append(angle_err_deg(root_f, gp[:, 0]).mean().item())
                acc['b_root_deg'].append(angle_err_deg(c['pose'][:, 0].to(device), gp[:, 0]).mean().item())
                acc['hk_deg'].append(angle_err_deg(hk_eval, gp[:, HIP_KNEE_JOINTS]).mean().item())
                acc['b_hk_deg'].append(angle_err_deg(c['pose'][:, HIP_KNEE_JOINTS].to(device), gp[:, HIP_KNEE_JOINTS]).mean().item())
        for m in mods:
            m.train()
        return {k: sum(v) / len(v) for k, v in acc.items()}

    history = [{'epoch': 0, **run_holdout()}]
    print(json.dumps(history[-1]), flush=True)
    for epoch in range(args.epochs):
        order = list(range(len(cached['train'])))
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        tot = []
        for idx in order:
            loss, parts = losses(mods, cached['train'][idx], device, args.mode, bodymodel, rms_deg=random.uniform(1.0, 8.0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot.append(loss.item())
        history.append({'epoch': epoch + 1, 'train_loss': sum(tot) / len(tot), **run_holdout()})
        print(json.dumps(history[-1]), flush=True)

    names = ['cam', 'ori', 'swing']
    for n, m in zip(names, mods):
        torch.save({'state_dict': m.state_dict(), 'cam_state_dict': m.state_dict() if n == 'cam' else None,
                    'manifest': {'hidden': 32, 'mode': args.mode, 'history': history}},
                   args.output / f'{n}.pt')
    (args.output / 'history.json').write_text(json.dumps(history, indent=2))
    print('Saved', args.output)


if __name__ == '__main__':
    main()
