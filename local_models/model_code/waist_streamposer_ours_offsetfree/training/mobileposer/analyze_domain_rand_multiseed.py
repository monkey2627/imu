"""Multi-seed robustness check for the domain-randomized independent-vs-joint
finding (2026-09-29): under time-constant domain randomization, JOINT beat
INDEPENDENT on BOTH axes at all 9 evaluated time-constant points for seed=0
-- a striking reversal of the original single-tc-trained boundary-condition
curve (see AnchorPose方案与验证结果_20260925.md). Given how surprising this
reversal was, re-check it against seed=1/2 (fresh --time-constant-range
0.05-6.0 training runs of both scripts) before treating it as a real
finding, following the same discipline already applied once to the original
boundary curve.

Run from SparseIMU/ after runtime_{indep,joint}_tcrand[_seed{1,2}]
checkpoints exist:
    python -m mobileposer.analyze_domain_rand_multiseed \
        --tran-cache /tmp/tran_cache.pt --ori-cache /tmp/ori_cache.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from mobileposer.synthetic_slam import synthesize_slam_joint, stable_seed
from mobileposer.camera_fusion import CameraFusionGRU
from mobileposer.orientation_fusion import OrientationFusionGRU
from mobileposer.joint_anchor_fusion import JointAnchorFusion
from mobileposer.train_joint_anchor_fusion import merge_caches, angle_between_rotmats

TC_POINTS = [0.1, 0.2, 0.3, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]
SEEDS = [0, 1, 2]


def indep_dir(seed):
    return 'runtime_indep_tcrand' if seed == 0 else f'runtime_indep_tcrand_seed{seed}'


def joint_dir(seed):
    return 'runtime_joint_tcrand' if seed == 0 else f'runtime_joint_tcrand_seed{seed}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tran-cache', type=Path, required=True)
    ap.add_argument('--ori-cache', type=Path, required=True)
    ap.add_argument('--slam-rms-m', type=float, default=0.66)
    ap.add_argument('--slam-rms-deg', type=float, default=3.0)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tran_blob = torch.load(args.tran_cache, map_location='cpu', weights_only=False)
    ori_blob = torch.load(args.ori_cache, map_location='cpu', weights_only=False)
    merged = merge_caches(tran_blob['cached'], ori_blob['cached'], 'holdout')
    print(json.dumps({'holdout_sequences': len(merged)}))

    results = {}  # results[tc][seed] = (pos_improve, pos_p, ori_improve, ori_p)
    for seed in SEEDS:
        idir, jdir = indep_dir(seed), joint_dir(seed)
        indep_blob = torch.load(Path(idir) / 'last.pt', map_location=device, weights_only=False)
        cam = CameraFusionGRU().to(device); cam.load_state_dict(indep_blob['cam_state_dict']); cam.eval()
        ori_net = OrientationFusionGRU().to(device); ori_net.load_state_dict(indep_blob['ori_state_dict']); ori_net.eval()
        joint_blob = torch.load(Path(jdir) / 'last.pt', map_location=device, weights_only=False)
        joint = JointAnchorFusion(hidden=40).to(device); joint.load_state_dict(joint_blob['state_dict']); joint.eval()

        for tc in TC_POINTS:
            indep_pos, joint_pos, indep_ori, joint_ori = [], [], [], []
            with torch.no_grad():
                for subset, source, tran_pred, tran_gt, ori_pred, ori_gt in merged:
                    s = stable_seed(source, 0)
                    slam_pos, slam_ori, _ = synthesize_slam_joint(
                        tran_gt, ori_gt, fps=30.0, target_rms_m=args.slam_rms_m,
                        target_rms_deg=args.slam_rms_deg, seed=s,
                        pos_time_constant_s=tc, rot_time_constant_s=tc)
                    imu_pos_b, slam_pos_b = tran_pred.unsqueeze(0).to(device), slam_pos.unsqueeze(0).to(device)
                    imu_ori_b, slam_ori_b = ori_pred.unsqueeze(0).to(device), slam_ori.unsqueeze(0).to(device)

                    cam_fused, _, _ = cam(imu_pos_b, slam_pos_b)
                    ori_fused, _, _ = ori_net(imu_ori_b, slam_ori_b)
                    indep_pos.append((cam_fused.squeeze(0).cpu() - tran_gt).norm(dim=-1).mean().item())
                    indep_ori.append(angle_between_rotmats(ori_fused.squeeze(0).cpu(), ori_gt).mean().item())

                    j_pos, j_ori, _, _, _, _ = joint(imu_pos_b, slam_pos_b, imu_ori_b, slam_ori_b)
                    joint_pos.append((j_pos.squeeze(0).cpu() - tran_gt).norm(dim=-1).mean().item())
                    joint_ori.append(angle_between_rotmats(j_ori.squeeze(0).cpu(), ori_gt).mean().item())

            indep_pos, joint_pos = np.array(indep_pos), np.array(joint_pos)
            indep_ori, joint_ori = np.array(indep_ori), np.array(joint_ori)
            pos_improve = 100 * (indep_pos.mean() - joint_pos.mean()) / indep_pos.mean()
            ori_improve = 100 * (indep_ori.mean() - joint_ori.mean()) / indep_ori.mean()
            pos_diff, ori_diff = indep_pos - joint_pos, indep_ori - joint_ori
            _, pos_p = wilcoxon(pos_diff) if not np.allclose(pos_diff, 0) else (None, 1.0)
            _, ori_p = wilcoxon(ori_diff) if not np.allclose(ori_diff, 0) else (None, 1.0)
            results.setdefault(tc, {})[seed] = (pos_improve, pos_p, ori_improve, ori_p)
            sig_pos = '*' if pos_p < 0.05 else ' '
            sig_ori = '*' if ori_p < 0.05 else ' '
            print(f'tc={tc:>4.1f} seed={seed}  pos_improve={pos_improve:>+7.2f}%{sig_pos} p={pos_p:.2e}  |  '
                  f'ori_improve={ori_improve:>+7.2f}%{sig_ori} p={ori_p:.2e}', flush=True)

    print('\n=== Sign/significance consistency across 3 seeds ===')
    for tc in TC_POINTS:
        rows = results[tc]
        if len(rows) < 3:
            print(f'tc={tc}: incomplete'); continue
        pos_signs = [np.sign(rows[s][0]) for s in SEEDS]
        ori_signs = [np.sign(rows[s][2]) for s in SEEDS]
        pos_sig_all = all(rows[s][1] < 0.05 for s in SEEDS)
        ori_sig_all = all(rows[s][3] < 0.05 for s in SEEDS)
        print(f'tc={tc:>4.1f}  pos: signs={pos_signs} all_significant={pos_sig_all}  |  '
              f'ori: signs={ori_signs} all_significant={ori_sig_all}')

    Path('runtime_domain_rand_multiseed_results.json').write_text(
        json.dumps({str(tc): {str(s): v for s, v in rows.items()} for tc, rows in results.items()}, indent=2))
    print('\nSaved runtime_domain_rand_multiseed_results.json')


if __name__ == '__main__':
    main()
