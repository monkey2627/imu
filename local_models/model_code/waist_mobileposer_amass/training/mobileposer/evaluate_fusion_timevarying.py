"""Learned vs hand-set fusion when SLAM quality is TIME-VARYING.

evaluate_fusion_baselines.py showed that with stationary external noise a single
fixed gain matches/beats the learned weight. The remaining case where a learned
per-frame weight could win is time-varying quality (tracking-loss / feature-poor
spells), where no single fixed gain is optimal. Noise: synthetic_slam.synthesize_slam_joint
(shared 'quality' latent multiplies position and heading error, scale clamp [0.3,4]).

Learned candidates:
  recal / dr           : trained on STATIONARY OU noise (the unified-pipeline defaults)
  indep_tcrand/tc03    : trained on THIS time-varying noise (matched), cam_state_dict / ori_state_dict
Hand-set: constant-gain complementary filter (translation), fixed-weight SLERP (orientation).

Run from SparseIMU/:  python -m mobileposer.evaluate_fusion_timevarying
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
from mobileposer.orientation_fusion import OrientationFusionGRU
from mobileposer.evaluate_drift_anchorpose import prepare_imu, LAYOUT_NAME
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from mobileposer.synthetic_slam import synthesize_slam_joint, stable_seed
from mobileposer.evaluate_fusion_baselines import slerp_toward, cf_translation
from drift_eval_common import load_long_sequences, angle_between_rotmats, FPS

SLERP_W = [0.1, 0.3, 0.5, 0.8]
CF_K = [0.01, 0.03, 0.1, 0.3]


def load(cls, path, key, dev):
    b = torch.load(path, map_location=dev, weights_only=False)
    m = cls(hidden=int(b.get('manifest', {}).get('hidden', 32))).to(dev)
    m.load_state_dict(b[key] if key in b else b['state_dict'])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_seqs', type=int, default=30)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    pipe = AnchorPosePipeline(device=args.device)
    dev = pipe.device
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)
    seqs = load_long_sequences(min_frames=1800, max_seqs=args.max_seqs)

    cams = {'learned_recal(stationary)': load(CameraFusionGRU, 'runtime_camera_fusion_recal/last.pt', 'cam_state_dict', dev),
            'learned_tcrand(matched)': load(CameraFusionGRU, 'runtime_indep_tcrand/last.pt', 'cam_state_dict', dev),
            'learned_tc03(matched)': load(CameraFusionGRU, 'runtime_indep_tc03/last.pt', 'cam_state_dict', dev)}
    oris = {'learned_dr(stationary)': load(OrientationFusionGRU, 'runtime_orientation_fusion_dr/last.pt', 'state_dict', dev),
            'learned_tcrand(matched)': load(OrientationFusionGRU, 'runtime_indep_tcrand/last.pt', 'ori_state_dict', dev),
            'learned_tc03(matched)': load(OrientationFusionGRU, 'runtime_indep_tc03/last.pt', 'ori_state_dict', dev)}

    cache = []
    for s in seqs:
        gp, gt = s['pose'].clone(), s['tran'].clone()
        ori, _, acc = _synthesize(pipe.bodymodel, gp, torch.ones(10), gt, LAYOUT_NAME, syn, 128)
        out = pipe(prepare_imu(acc, ori), None, None, use_cam=False, use_ori=False, use_swing=False)
        T = len(out['base_tran'])
        cache.append({'src': s['source'], 'root': out['base_pose'][:, 0], 'gt_root': gp[:T, 0], 'tran': out['base_tran'],
                      'gt_tran': gt[:T], 'gt_rel': (gt - gt[:1])[:T]})
    print(f'cached {len(cache)}', flush=True)

    scenarios = [('good(0.05m,0.6deg)', 0.05, 0.6, 1.5), ('mid(0.2m,1.5deg)', 0.2, 1.5, 1.5),
                 ('cons(0.66m,3deg)', 0.66, 3.0, 1.5), ('good_burst(0.05m,0.6deg,q3)', 0.05, 0.6, 3.0),
                 ('mid_burst(0.2m,1.5deg,q3)', 0.2, 1.5, 3.0)]
    t_methods = ['imu_only', 'slam_only'] + [f'cf_k{k}' for k in CF_K] + list(cams)
    o_methods = ['imu_only', 'slam_only'] + [f'slerp_w{w}' for w in SLERP_W] + list(oris)
    res_t, res_o = {}, {}
    for name, rm, rd, qs in scenarios:
        et = {m: [] for m in t_methods}
        eo = {m: [] for m in o_methods}
        for c in cache:
            sp, so, _ = synthesize_slam_joint(c['gt_tran'], c['gt_root'], fps=FPS, target_rms_m=rm, target_rms_deg=rd,
                                              quality_strength=qs, seed=stable_seed(c['src'], offset=21))
            sp = sp - c['gt_tran'][:1]
            imu_t, imu_r = c['tran'], c['root']
            outs_t = {'imu_only': imu_t, 'slam_only': sp}
            for k in CF_K:
                outs_t[f'cf_k{k}'] = cf_translation(imu_t, sp, k)
            outs_o = {'imu_only': imu_r, 'slam_only': so}
            for w in SLERP_W:
                outs_o[f'slerp_w{w}'] = slerp_toward(imu_r, so, w)
            with torch.no_grad():
                for n, m in cams.items():
                    outs_t[n] = m(imu_t.unsqueeze(0).to(dev), sp.unsqueeze(0).to(dev))[0].squeeze(0).cpu()
                for n, m in oris.items():
                    outs_o[n] = m(imu_r.unsqueeze(0).to(dev), so.unsqueeze(0).to(dev))[0].squeeze(0).cpu()
            for m in t_methods:
                et[m].append((outs_t[m] - c['gt_rel']).norm(dim=-1).mean().item())
            for m in o_methods:
                eo[m].append(angle_between_rotmats(outs_o[m], c['gt_root']).mean().item())
        res_t[name] = {m: float(np.mean(v)) for m, v in et.items()}
        res_o[name] = {m: float(np.mean(v)) for m, v in eo.items()}
        print(f'  {name} done', flush=True)

    for title, res, methods, unit, fmt in (('TRANSLATION (m)', res_t, t_methods, 'm', '{:12.3f}'),
                                            ('ORIENTATION (deg)', res_o, o_methods, 'deg', '{:12.2f}')):
        print(f'\n=== {title}, time-varying SLAM quality, mean over 30 seqs ===')
        print('scenario'.ljust(30) + ''.join(m[:12].rjust(13) for m in methods))
        for sc, r in res.items():
            print(sc.ljust(30) + ''.join(fmt.format(r[m]).rjust(13) for m in methods))
        learned = [m for m in methods if m.startswith('learned')]
        fixed = [m for m in methods if m.startswith('cf') or m.startswith('slerp')]
        print('mean over scenarios: ' + '  '.join(f'{m}={np.mean([r[m] for r in res.values()]):.3f}' for m in methods))
        for lm in learned:
            wins = sum(r[lm] < min(r[f] for f in fixed + ['imu_only', 'slam_only']) for r in res.values())
            print(f'  {lm}: beats EVERY fixed policy (incl imu_only/slam_only) in {wins}/{len(res)} scenarios')


if __name__ == '__main__':
    main()
