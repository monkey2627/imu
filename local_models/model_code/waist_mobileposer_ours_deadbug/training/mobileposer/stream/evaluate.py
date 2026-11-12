"""Standard-protocol evaluation of the streaming model (DIP / TC-DIP calib / TC official calib; SIP/Ang/Pos/Mesh).

The whole sequence is streamed from a zero state (no GT or T-pose initialisation); the pose of frame t is read from step
t+K, i.e. it uses IMU up to t+K (K = training lag; 5 frames at 60 fps = 83 ms).  The end is padded by repeating the last
frame.  --window W limits the history to W frames (stateless sliding windows) to measure what the long state is worth.

  MOBILEPOSER_FPS=60 python -m mobileposer.stream.evaluate --name stream_k5_s42
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import mobileposer.evaluate_standard as es
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.stream.features import imu_features, to_local_pose, root_relative_ori
from mobileposer.stream.model import StreamPoser

CK = Path('checkpoints_research')
NAMES = ['pelvis', 'L hip', 'R hip', 'spine1', 'L knee', 'R knee', 'spine2', 'L ankle', 'R ankle', 'spine3', 'L foot',
         'R foot', 'neck', 'L collar', 'R collar', 'head', 'L shoulder', 'R shoulder', 'L elbow', 'R elbow']


@torch.no_grad()
def predict(model, a, R, K, gyro, window=0, bs=512):
    """a [T,5,3], R [T,5,3,3] (on device) -> reduced r6d [T,90] for frames 0..T-1"""
    T = len(a)
    a = torch.cat([a, a[-1:].expand(K, -1, -1)]); R = torch.cat([R, R[-1:].expand(K, -1, -1, -1)])
    x = imu_features(a[None], R[None], gyro, model.frame, model.local)[0]          # [T+K,F]
    r6d = model(x[None])[1][0, K:K + T]
    if window:
        L = window + K + 1                                                          # frames t-window .. t+K
        out = [r6d[:window]]                                                         # t < window: same as streaming from 0
        for s in range(window, T, bs):
            ts = torch.arange(s, min(T, s + bs), device=x.device)
            idx = (ts - window)[:, None] + torch.arange(L, device=x.device)[None]
            # features are recomputed per window so the angular velocity of its first frame is zero, like a cold start
            xa = imu_features(a[idx], R[idx], gyro, model.frame, model.local)
            out.append(model(xa)[1][:, -1])
        r6d = torch.cat(out)
    return model.compose(r6d[None], root_relative_ori(R[None, :T]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--window', type=int, default=0)
    ap.add_argument('--datasets', default='dipimu,totalcapture_dipcalib,totalcapture_officalib')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    assert es.FPS == 60, 'set MOBILEPOSER_FPS=60'
    dev = torch.device('cuda:0')
    cfg = json.load(open(CK / args.name / 'config.json'))
    K, gyro = cfg['lag'], bool(cfg['gyro'])
    model = StreamPoser(gyro, cfg['hidden'], cfg['layers'], anchor=cfg.get('anchor', 'none'), frame=cfg.get('frame', 'rel'), arch=cfg.get('arch', 'mp2'),
                        local=bool(cfg.get('local_gyro', 0))).to(dev).eval()
    model.load_state_dict(torch.load(CK / args.name / 'model.pt', map_location=dev))
    body = ParametricModel(paths.smpl_file, device=dev)
    ev = es.StdEvaluator(dev)
    art = ev.art
    pnp_body = art.ParametricModel(str(es.PNP_DIR / 'models' / 'SMPL_male.pkl'), device=dev)
    ign = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
    tag = args.name + (f'_w{args.window}' if args.window else '') + args.tag
    res = {'name': args.name, 'lag_frames': K, 'lookahead_ms': 1000 * K / 60, 'window': args.window, 'init': 'zero state'}
    out_dir = Path('results/stream'); out_dir.mkdir(parents=True, exist_ok=True)
    checked = False
    for ds in args.datasets.split(','):
        errs, perj, w, frames = [], [], [], {}
        for name, a, R, pose in es.sequences(ds, art):
            a, R = a.to(dev), R.to(dev)
            r6d = predict(model, a, R, K, gyro, args.window)
            if not checked:                      # causality: frames after t+K must not change the output of frame t
                t0 = min(300, len(a) // 2)
                a2, R2 = a.clone(), R.clone()
                a2[t0 + K + 1:] = torch.randn_like(a2[t0 + K + 1:]) * 5
                R2[t0 + K + 1:] = R2[t0 + K + 1:].flip(0)
                r2 = predict(model, a2, R2, K, gyro, args.window)
                res['future_invariance_max_abs_error'] = float((r2[:t0 + 1] - r6d[:t0 + 1]).abs().max())
                print('future invariance max abs error', res['future_invariance_max_abs_error'], flush=True)
                checked = True
            p = to_local_pose(body, r6d, R[:, 4]).cpu()
            errs.append(ev(p, pose))
            pp, qq = p.clone().to(dev), pose.clone().to(dev)
            pp[:, ign] = torch.eye(3, device=dev); qq[:, ign] = torch.eye(3, device=dev)
            Gp = pnp_body.forward_kinematics(pp, calc_mesh=False)[0].view(-1, 24, 3, 3)
            Gq = pnp_body.forward_kinematics(qq, calc_mesh=False)[0].view(-1, 24, 3, 3)
            ang = art.math.radian_to_degree(art.math.angle_between(Gp, Gq).view(-1, 24)).cpu().numpy()
            perj.append(ang.mean(0)); w.append(len(ang)); frames[str(name)] = ang[:, [1, 2, 16, 17]]
            print('  ', ds, name, np.round(errs[-1].numpy(), 2), flush=True)
        res[ds] = torch.stack(errs).mean(0).tolist()
        pj = np.average(perj, axis=0, weights=w)
        res[ds + '_perjoint'] = {NAMES[k]: float(pj[k]) for k in range(20)}
        np.savez_compressed(out_dir / f'{tag}_{ds}_frames.npz', **frames)
        print(ds, 'SIP/Ang/Pos/Mesh', np.round(res[ds], 2), '| hips %.1f shoulders %.1f' % (pj[[1, 2]].mean(), pj[[16, 17]].mean()), flush=True)
    json.dump(res, open(out_dir / f'{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
