"""Why does runtime_indep_tcrand (time-constant domain-randomised) make
translation WORSE at the calibrated tc=0.3 s while runtime_camera_fusion_recal
(trained at tc=0.3 s) improves it? Sweep the test-time SLAM time constant and
compare mean translation error of: IMU-only, each camera-fusion checkpoint.

Run from SparseIMU/:  python -m mobileposer.diagnose_cam_ckpt_tc
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.anchorpose_pipeline import AnchorPosePipeline
from mobileposer.camera_fusion import CameraFusionGRU
from mobileposer.evaluate_drift_anchorpose import prepare_imu, LAYOUT_NAME
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from mobileposer.synthetic_slam import synthesize_slam_position, stable_seed
from drift_eval_common import load_long_sequences, FPS

CKPTS = ['runtime_camera_fusion_recal', 'runtime_indep_tcrand', 'runtime_indep_tc03', 'runtime_indep_tc10']
TCS = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]


def load_cam(path, dev):
    blob = torch.load(path, map_location=dev, weights_only=False)
    hidden = int(blob.get('manifest', {}).get('hidden', 32))
    m = CameraFusionGRU(hidden=hidden).to(dev)
    m.load_state_dict(blob['cam_state_dict'] if 'cam_state_dict' in blob else blob['state_dict'])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_seqs', type=int, default=30)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    pipe = AnchorPosePipeline(device=args.device)
    dev = pipe.device
    cams = {c: load_cam(f'{c}/last.pt', dev) for c in CKPTS}
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)
    seqs = load_long_sequences(min_frames=1800, max_seqs=args.max_seqs)

    err = {(c, tc): [] for c in ['imu'] + CKPTS for tc in TCS}
    for s in seqs:
        gp, gt = s['pose'].clone(), s['tran'].clone()
        ori, _, a = _synthesize(pipe.bodymodel, gp, torch.ones(10), gt, LAYOUT_NAME, syn, 128)
        out = pipe(prepare_imu(a, ori), None, None, use_cam=False, use_ori=False, use_swing=False)
        tran = out['tran']
        rel = (gt - gt[:1])[:len(tran)]
        for tc in TCS:
            sp = synthesize_slam_position(gt, fps=FPS, target_rms_m=0.66, time_constant_s=tc, seed=stable_seed(s['source'])) - gt[:1]
            err[('imu', tc)].append((tran - rel).norm(dim=-1).mean().item())
            for c, cam in cams.items():
                with torch.no_grad():
                    f, _, _ = cam(tran.unsqueeze(0).to(dev), sp[:len(tran)].unsqueeze(0).to(dev))
                err[(c, tc)].append((f.squeeze(0).cpu() - rel).norm(dim=-1).mean().item())
        print(f'  {s["source"]} done', flush=True)

    print('\nmean translation error (m) vs test SLAM time constant (s); SLAM-only error ~0.66 m RMS')
    print('ckpt'.ljust(30) + ''.join(f'tc={tc:<7}' for tc in TCS))
    for c in ['imu'] + CKPTS:
        print(c.ljust(30) + ''.join(f'{np.mean(err[(c, tc)]):<10.3f}' for tc in TCS))


if __name__ == '__main__':
    main()
