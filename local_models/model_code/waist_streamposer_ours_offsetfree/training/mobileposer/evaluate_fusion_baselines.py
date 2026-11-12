"""Does the LEARNED per-frame weight beat simple hand-set fusion?  (novelty check
for the AnchorPose external-anchor modules; nearest prior art uses hand-set
weights: EgoLocate fixed SLERP 0.1, RobustCap confidence formula, Kalman gains).

Root ORIENTATION (OrientationFusionGRU vs fixed-weight SLERP vs gated SLERP) and
root TRANSLATION (CameraFusionGRU vs constant-gain complementary filter), on the
30 held-out long AMASS sequences, over a grid of scenarios:
  orientation: external-heading noise RMS in {0.6,1.5,3,6,8} deg (OU, tc 0.4 s),
               + a 'spikes' scenario (3 deg OU + ~2% of frames in 1 s bursts of 11.8 deg),
               x injected IMU yaw drift in {0, 0.03, 0.1} deg/s (left-multiplied Ry; the
               synthetic IMU data has no sensor drift, so without injection the anchor can
               only fix the network's own ~4 deg error).
  translation: SLAM RMS in {0.05,0.2,0.66,1.5} m at tc 0.3 s, and 0.66 m at tc 2.0 s.
Reported: mean error per method per scenario, plus a summary of how each fixed
weight ranks across ALL scenarios (a single fixed gain must work everywhere).

Run from SparseIMU/:  python -m mobileposer.evaluate_fusion_baselines
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.anchorpose_pipeline import AnchorPosePipeline
from mobileposer.evaluate_drift_anchorpose import prepare_imu, LAYOUT_NAME
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from mobileposer.synthetic_slam import synthesize_slam_position, synthesize_slam_orientation, stable_seed
from mobileposer.orientation_fusion import _so3_log, _so3_exp
from drift_eval_common import load_long_sequences, angle_between_rotmats, FPS

FIXED_W = [0.02, 0.05, 0.1, 0.3, 0.5]
FIXED_K = [0.01, 0.03, 0.1, 0.3]


def slerp_toward(imu, ext, w):
    rel = _so3_log(imu.transpose(-1, -2) @ ext)
    w = w if torch.is_tensor(w) else torch.tensor(w)
    return imu @ _so3_exp(rel * (w.unsqueeze(-1) if w.dim() else w))


def gated_slerp(imu, ext, w, thr_deg):
    disc = angle_between_rotmats(imu, ext)
    wv = torch.where(disc <= thr_deg, torch.tensor(float(w)), torch.tensor(0.0))
    return slerp_toward(imu, ext, wv)


def ry(theta):
    """theta [T] radians -> [T,3,3] rotation about Y (up axis of the training basis)."""
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.zeros(len(theta), 3, 3)
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1.0, -s, c
    return R


def add_spikes(slam_ori, seed, frac=0.02, burst=30, deg=11.8):
    g = torch.Generator().manual_seed(seed)
    T = len(slam_ori)
    out = slam_ori.clone()
    n_bursts = max(1, int(frac * T / burst))
    for _ in range(n_bursts):
        s = int(torch.randint(0, max(1, T - burst), (1,), generator=g))
        axis = torch.randn(3, generator=g)
        axis = axis / axis.norm()
        R = _so3_exp(axis * math.radians(deg))
        out[s:s + burst] = R @ out[s:s + burst]
    return out


def cf_translation(imu, slam, k):
    """constant-gain complementary filter on translation increments."""
    out = torch.zeros_like(imu)
    out[0] = imu[0]
    for t in range(1, len(imu)):
        pred = out[t - 1] + (imu[t] - imu[t - 1])
        out[t] = pred + k * (slam[t] - pred)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_seqs', type=int, default=30)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    pipe = AnchorPosePipeline(device=args.device)
    dev = pipe.device
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)
    seqs = load_long_sequences(min_frames=1800, max_seqs=args.max_seqs)

    cache = []
    for s in seqs:
        gp, gt = s['pose'].clone(), s['tran'].clone()
        ori, _, acc = _synthesize(pipe.bodymodel, gp, torch.ones(10), gt, LAYOUT_NAME, syn, 128)
        out = pipe(prepare_imu(acc, ori), None, None, use_cam=False, use_ori=False, use_swing=False)
        T = len(out['base_tran'])
        cache.append({'src': s['source'], 'root': out['base_pose'][:, 0], 'gt_root': gp[:T, 0],
                      'tran': out['base_tran'], 'gt_tran': gt[:T], 'gt_rel': (gt - gt[:1])[:T]})
    print(f'cached {len(cache)} sequences', flush=True)

    # ------------------------------------------------------------------ orientation
    ori_methods = ['imu_only', 'slam_only'] + [f'slerp_w{w}' for w in FIXED_W] + ['gated_w0.1@10deg', 'learned']
    scen_ori = [(f'rms{r}', r, False) for r in (0.6, 1.5, 3.0, 6.0, 8.0)] + [('spikes3', 3.0, True)]
    drifts = (0.0, 0.03, 0.1)
    res_o = {}
    for sc, rms, spikes in scen_ori:
        for dr in drifts:
            errs = {m: [] for m in ori_methods}
            for i, c in enumerate(cache):
                T = len(c['root'])
                seed = stable_seed(c['src'], offset=7)
                slam = synthesize_slam_orientation(c['gt_root'], fps=FPS, target_rms_deg=rms, seed=seed)
                if spikes:
                    slam = add_spikes(slam, seed)
                imu = c['root']
                if dr > 0:
                    sign = 1.0 if (stable_seed(c['src'], offset=3) % 2 == 0) else -1.0
                    theta = torch.deg2rad(torch.arange(T, dtype=torch.float32) / FPS * dr * sign)
                    imu = ry(theta) @ imu
                gt = c['gt_root']
                outs = {'imu_only': imu, 'slam_only': slam}
                for w in FIXED_W:
                    outs[f'slerp_w{w}'] = slerp_toward(imu, slam, w)
                outs['gated_w0.1@10deg'] = gated_slerp(imu, slam, 0.1, 10.0)
                with torch.no_grad():
                    f, _, _ = pipe.ori(imu.unsqueeze(0).to(dev), slam.unsqueeze(0).to(dev))
                outs['learned'] = f.squeeze(0).cpu()
                for m in ori_methods:
                    errs[m].append(angle_between_rotmats(outs[m], gt).mean().item())
            res_o[(sc, dr)] = {m: float(np.mean(v)) for m, v in errs.items()}
        print(f'  orientation scenario {sc} done', flush=True)

    print('\n=== ROOT ORIENTATION error (deg), mean over 30 seqs ===')
    print('scenario(ext noise, imu drift deg/s)'.ljust(34) + ''.join(m.rjust(18) for m in ori_methods))
    for (sc, dr), r in res_o.items():
        print(f'{sc:>8s} drift={dr:<5}'.ljust(34) + ''.join(f'{r[m]:18.2f}' for m in ori_methods))
    fixed_names = [m for m in ori_methods if m.startswith('slerp') or m.startswith('gated')]
    print('\nrank summary (orientation): mean error over all scenarios; worst-case scenario error')
    for m in ori_methods:
        v = np.array([r[m] for r in res_o.values()])
        print(f'  {m:20s} mean={v.mean():6.2f}  worst={v.max():6.2f}')
    best_fixed_hindsight = np.mean([min(r[m] for m in fixed_names) for r in res_o.values()])
    best_single_fixed = min(fixed_names, key=lambda m: np.mean([r[m] for r in res_o.values()]))
    print(f'  best single fixed/gated method over all scenarios: {best_single_fixed} '
          f'(mean {np.mean([r[best_single_fixed] for r in res_o.values()]):.2f}); '
          f'per-scenario best-in-hindsight fixed: mean {best_fixed_hindsight:.2f}')
    n_win = sum(r['learned'] <= min(r[m] for m in fixed_names) + 1e-9 for r in res_o.values())
    print(f'  learned <= best fixed in hindsight in {n_win}/{len(res_o)} scenarios')

    # ------------------------------------------------------------------ translation
    tr_methods = ['imu_only', 'slam_only'] + [f'cf_k{k}' for k in FIXED_K] + ['learned']
    scen_tr = [(f'rms{r}_tc0.3', r, 0.3) for r in (0.05, 0.2, 0.66, 1.5)] + [('rms0.66_tc2.0', 0.66, 2.0)]
    res_t = {}
    for sc, rms, tc in scen_tr:
        errs = {m: [] for m in tr_methods}
        for c in cache:
            slam = synthesize_slam_position(c['gt_tran'], fps=FPS, target_rms_m=rms, time_constant_s=tc,
                                            seed=stable_seed(c['src'], offset=11)) - c['gt_tran'][:1]
            imu = c['tran']
            rel = c['gt_rel']
            outs = {'imu_only': imu, 'slam_only': slam}
            for k in FIXED_K:
                outs[f'cf_k{k}'] = cf_translation(imu, slam, k)
            with torch.no_grad():
                f, _, _ = pipe.cam(imu.unsqueeze(0).to(dev), slam.unsqueeze(0).to(dev))
            outs['learned'] = f.squeeze(0).cpu()
            for m in tr_methods:
                errs[m].append((outs[m] - rel).norm(dim=-1).mean().item())
        res_t[sc] = {m: float(np.mean(v)) for m, v in errs.items()}
        print(f'  translation scenario {sc} done', flush=True)

    print('\n=== ROOT TRANSLATION error (m), mean over 30 seqs ===')
    print('scenario'.ljust(18) + ''.join(m.rjust(14) for m in tr_methods))
    for sc, r in res_t.items():
        print(sc.ljust(18) + ''.join(f'{r[m]:14.3f}' for m in tr_methods))
    cf_names = [m for m in tr_methods if m.startswith('cf')]
    best_single_cf = min(cf_names, key=lambda m: np.mean([r[m] for r in res_t.values()]))
    print(f'  best single constant gain over all scenarios: {best_single_cf} '
          f'(mean {np.mean([r[best_single_cf] for r in res_t.values()]):.3f}); '
          f'learned mean {np.mean([r["learned"] for r in res_t.values()]):.3f}; '
          f'imu_only mean {np.mean([r["imu_only"] for r in res_t.values()]):.3f}')
    n_win = sum(r['learned'] <= min(r[m] for m in cf_names) + 1e-9 for r in res_t.values())
    print(f'  learned <= best constant gain in hindsight in {n_win}/{len(res_t)} scenarios')


if __name__ == '__main__':
    main()
