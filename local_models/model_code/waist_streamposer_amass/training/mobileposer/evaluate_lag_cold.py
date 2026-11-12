"""Fixed-lag causal outputs vs cold start on the 68 UNSEEN stand->hold events.

A lag-L model (trained with MOBILEPOSER_LAG=L) outputs the pose of frame t when it has seen the IMU
up to frame t+L, i.e. a real-time system with latency L/30 s.  L=0 is the plain causal model.
Conditions (thigh error, deg, unaligned global, time bins since the read-out start):
  history : model started at the sequence start (it saw the standing phase); read from COLD point
  cold    : model started at COLD = t0 + 90 frames (already inside the squat)
Also reports the bidirectional reference.

Run from SparseIMU/:  python -m mobileposer.evaluate_lag_cold
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_v2_models import ModulePredictor
from mobileposer.evaluate_causal_holds import BINS
from drift_eval_common import angle_between_rotmats

COLD = 90


class LagPredictor(ModulePredictor):
    """pose[t] = output at step t+lag, expressed in the waist frame of step t+lag"""

    def __init__(self, poser, joints, causal, lag, dev):
        if causal:
            os.environ['MOBILEPOSER_CAUSAL'] = '1'
        try:
            super().__init__(poser, joints, True, dev)
        finally:
            os.environ.pop('MOBILEPOSER_CAUSAL', None)
        self.lag = lag

    def __call__(self, acc, ori):
        pose = super().__call__(acc, ori)          # root already uses Rw[k] at output step k
        T = len(pose)
        idx = (torch.arange(T) + self.lag).clamp(max=T - 1)
        return pose[idx]


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    models = {'bidir': LagPredictor('checkpoints/v3_canon_poser', 'checkpoints/v3_canon_joints', False, 0, dev),
              'causal lag0': LagPredictor('checkpoints/v3_causal125_poser', 'checkpoints/v3_causal125_joints', True, 0, dev)}
    for lag in (15, 30, 60):
        if Path(f'checkpoints/v3_lag{lag}_poser').exists() and Path(f'checkpoints/v3_lag{lag}_joints').exists():
            models[f'causal lag{lag} ({lag / 30:.1f}s)'] = LagPredictor(f'checkpoints/v3_lag{lag}_poser', f'checkpoints/v3_lag{lag}_joints', True, lag, dev)
    blob = torch.load('data/hold_events_heldout.pt', map_location='cpu', weights_only=False)
    rows = []
    for sub, i, t0, end, T in blob['events']:
        h = blob['held'][sub]
        j = h['indices'].index(i)
        acc, ori, pose = h['acc'][j], h['ori'][j], h['pose'][j].reshape(-1, 24, 3, 3)
        T = min(len(pose), 3000)
        acc, ori, pose = acc[:T], ori[:T], pose[:T]
        s, end = t0 + COLD, min(end, T)
        if end - s < 150:
            continue
        G, _ = body.forward_kinematics(pose, torch.ones(10), tran=None, calc_mesh=False)
        G = G.reshape(-1, 24, 3, 3)
        rel = G[:, 3].transpose(-1, -2) @ G[:, 4]
        U, S, Vt = torch.linalg.svd(rel[max(0, t0 - 60):t0 - 10].mean(0))
        amb = angle_between_rotmats(rel[s:end], (U @ Vt).unsqueeze(0).expand(end - s, 3, 3)).mean().item()
        r = {'amb': amb}
        for m, fn in models.items():
            for name, a0 in (('history', 0), ('cold', s)):
                pred = fn(acc[a0:], ori[a0:])[s - a0:end - a0]
                Gp, _ = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
                terr = angle_between_rotmats(Gp.reshape(-1, 24, 3, 3)[:, 1], G[s:end, 1]).numpy()
                r[(m, name)] = [terr[int(lo * 30):min(end - s, int(hi * 30))].mean()
                                if min(end - s, int(hi * 30)) - int(lo * 30) >= 10 else np.nan for lo, hi in BINS]
        rows.append(r)
        print('  event', sub, i, flush=True)
    amb = np.array([r['amb'] for r in rows])
    keys = [k for k in rows[0] if isinstance(k, tuple)]
    print(f'\n{len(rows)} events')
    for name, sel in (('ALL', np.ones(len(rows), bool)), ('AMBIGUOUS (<30 deg)', amb < 30), ('clearly different (>=30 deg)', amb >= 30)):
        print(f'\n--- {name}: {sel.sum()} events ---   thigh error (deg) by time since cold-start point')
        print('model / input'.ljust(34) + '  '.join(f'{lo}-{hi if hi < 1000 else "end"}s'.rjust(8) for lo, hi in BINS) + '     mean')
        for k in keys:
            v = np.array([r[k] for r, x in zip(rows, sel) if x])
            print(f'{k[0]} / {k[1]}'.ljust(34) + '  '.join(f'{np.nanmean(v[:, b]):8.1f}' for b in range(len(BINS))) + f'  {np.nanmean(v):7.1f}')


if __name__ == '__main__':
    main()
