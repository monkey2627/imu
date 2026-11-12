"""COLD START on the 68 UNSEEN stand->hold events: what if the session begins already inside the hold?

For each event the model is run (a) from the sequence start (it has seen the standing phase) and
(b) from COLD = t0 + 90 frames (3 s after flexion onset: the person is already in the posture, no
standing history).  Thigh error (deg, unaligned global) is read on the SAME frames in the hold, in
time bins since COLD.  Models: bidir (v3_canon, full sequence) and causal125 (past only).

Run from SparseIMU/:  python -m mobileposer.evaluate_cold_start
"""
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_causal_holds import load, BINS
from drift_eval_common import angle_between_rotmats

COLD = 90


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    models = {'bidir(v3_canon)': load('checkpoints/v3_canon_poser', 'checkpoints/v3_canon_joints', False, dev),
              'causal125': load('checkpoints/v3_causal125_poser', 'checkpoints/v3_causal125_joints', True, dev)}
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
            for name, a0 in (('with history', 0), ('cold start', s)):
                pred = fn(acc[a0:], ori[a0:])[:T - a0][s - a0:]
                Gp, _ = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
                terr = angle_between_rotmats(Gp.reshape(-1, 24, 3, 3)[:, 1], G[s:s + len(pred), 1]).numpy()
                r[(m, name)] = [terr[int(lo * 30):min(end - s, int(hi * 30))].mean()
                                if min(end - s, int(hi * 30)) - int(lo * 30) >= 10 else np.nan for lo, hi in BINS]
        rows.append(r)
        print('  event', sub, i, flush=True)
    amb = np.array([r['amb'] for r in rows])
    keys = [k for k in rows[0] if isinstance(k, tuple)]
    print(f'\n{len(rows)} events with >=5 s of hold after the cold-start point; ambiguity quartiles {np.round(np.quantile(amb, [0, .25, .5, .75, 1]), 1)}')
    for name, sel in (('ALL', np.ones(len(rows), bool)), ('AMBIGUOUS (<30 deg)', amb < 30), ('clearly different (>=30 deg)', amb >= 30)):
        print(f'\n--- {name}: {sel.sum()} events ---   thigh error (deg) by time since cold-start point')
        print('model / input'.ljust(34) + '  '.join(f'{lo}-{hi if hi < 1000 else "end"}s'.rjust(8) for lo, hi in BINS))
        for k in keys:
            v = np.array([r[k] for r, x in zip(rows, sel) if x])
            print(f'{k[0]} / {k[1]}'.ljust(34) + '  '.join(f'{np.nanmean(v[:, b]):8.1f}' for b in range(len(BINS))))


if __name__ == '__main__':
    main()
