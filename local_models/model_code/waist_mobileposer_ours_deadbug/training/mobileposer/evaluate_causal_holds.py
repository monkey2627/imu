"""Causal (past-only) long memory on the 68 UNSEEN stand->hold events (see evaluate_heldout_holds.py).

Models (all canonical waist frame, Poser+Joints, trained on v3 = AMASS minus the 68 events):
  bidir (v3_canon)       : bidirectional LSTM, trained on 125-frame windows (reference, NOT causal)
  causal125              : unidirectional LSTM, trained on 125-frame windows
  causal1000             : unidirectional LSTM, trained on 1000-frame windows
Inference: hidden state reset every N frames (N = 46, 125, 1000, full sequence); a causal model with
N frames of reset context only uses the past N frames, i.e. it is real-time capable.
Thigh orientation error (deg, unaligned global) in time bins after flexion onset, split by standing-likeness.

Run from SparseIMU/:  python -m mobileposer.evaluate_causal_holds
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
from drift_eval_common import angle_between_rotmats

BINS = [(0, 4), (4, 8), (8, 16), (16, 1000)]
CHUNKS = [46, 125, 1000, 0]          # 0 = whole sequence


def load(poser, joints, causal, dev):
    if causal:
        os.environ['MOBILEPOSER_CAUSAL'] = '1'
    try:
        return ModulePredictor(poser, joints, True, dev)
    finally:
        os.environ.pop('MOBILEPOSER_CAUSAL', None)


def predict(fn, acc, ori, chunk):
    T = len(acc)
    if chunk == 0 or chunk >= T:
        return fn(acc, ori)[:T]
    return torch.cat([fn(acc[s:s + chunk], ori[s:s + chunk])[:min(chunk, T - s)] for s in range(0, T, chunk)])[:T]


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    models = {'bidir(v3_canon)': load('checkpoints/v3_canon_poser', 'checkpoints/v3_canon_joints', False, dev),
              'causal125': load('checkpoints/v3_causal125_poser', 'checkpoints/v3_causal125_joints', True, dev),
              'causal1000': load('checkpoints/v3_causal1000_poser', 'checkpoints/v3_causal1000_joints', True, dev)}
    blob = torch.load('data/hold_events_heldout.pt', map_location='cpu', weights_only=False)
    events, held = blob['events'], blob['held']
    rows = []
    for sub, i, t0, end, T in events:
        h = held[sub]
        j = h['indices'].index(i)
        acc, ori, pose = h['acc'][j], h['ori'][j], h['pose'][j].reshape(-1, 24, 3, 3)
        T = min(len(pose), 3000)
        acc, ori, pose = acc[:T], ori[:T], pose[:T]
        G, _ = body.forward_kinematics(pose, torch.ones(10), tran=None, calc_mesh=False)
        G = G.reshape(-1, 24, 3, 3)
        rel = G[:, 3].transpose(-1, -2) @ G[:, 4]
        pre, hold = rel[max(0, t0 - 60):t0 - 10], rel[t0 + 60:min(end, T)]
        U, S, Vt = torch.linalg.svd(pre.mean(0))
        amb = angle_between_rotmats(hold, (U @ Vt).unsqueeze(0).expand(len(hold), 3, 3)).mean().item()
        r = {'amb': amb}
        for m, fn in models.items():
            for chunk in CHUNKS:
                pred = predict(fn, acc, ori, chunk)
                Gp, _ = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
                terr = angle_between_rotmats(Gp.reshape(-1, 24, 3, 3)[:, 1], G[:, 1]).numpy()
                r[(m, chunk)] = [terr[t0 + int(lo * 30):min(end, t0 + int(hi * 30))].mean()
                                 if min(end, t0 + int(hi * 30)) - (t0 + int(lo * 30)) >= 10 else np.nan for lo, hi in BINS]
        rows.append(r)
        print('  event', sub, i, flush=True)
    amb = np.array([r['amb'] for r in rows])
    keys = [k for k in rows[0] if isinstance(k, tuple)]
    print(f'\n{len(rows)} held-out events; ambiguity quartiles {np.round(np.quantile(amb, [0, .25, .5, .75, 1]), 1)}')
    for name, sel in (('ALL', np.ones(len(rows), bool)), ('AMBIGUOUS (<30 deg, standing-like)', amb < 30), ('clearly different (>=30 deg)', amb >= 30)):
        print(f'\n--- {name}: {sel.sum()} events ---   thigh error (deg) by time since flexion onset')
        print('model / reset every N frames'.ljust(34) + '  '.join(f'{lo}-{hi if hi < 1000 else "end"}s'.rjust(8) for lo, hi in BINS))
        for k in keys:
            v = np.array([r[k] for r, s in zip(rows, sel) if s])
            print(f'{k[0]} / {k[1] or "full"}'.ljust(34) + '  '.join(f'{np.nanmean(v[:, b]):8.1f}' for b in range(len(BINS))))


if __name__ == '__main__':
    main()
