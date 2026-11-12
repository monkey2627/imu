"""History / memory test on UNSEEN stand->hold events.

68 AMASS sequences were found where the person goes from standing (knee flexion < 25 deg) to a
knee flexion >= 50 deg that is held >= 8 s (KIT 38, HUMAN4D 12, CMU 6, Eyes_Japan 6, HDM05 4, ...).
They were REMOVED from training (v3 data) and saved in data/hold_events_heldout.pt; v3_base / v3_canon
(Poser+Joints, 48 h minus these sequences) never saw them.  (Caveat: other trials of the same subject
and task stay in training - sequence-level, not subject-level, hold-out.)

For every event: thigh orientation error (deg, unaligned global) in time bins after the flexion onset,
split by how much the hold looks like standing (AMBIGUITY = angle between the hold-phase and the
pre-stand waist->shank relative orientation, small = standing-like) and by inference mode:
  full   : whole sequence (up to 3000 frames) in one pass
  win125 : hidden state reset every 125 frames (the training window length)
  win46  : real-time window, 40 past + 5 future frames
Compared with v2 (which saw them) to show the in-sample/held-out gap.

Run from SparseIMU/:  python -m mobileposer.evaluate_heldout_holds
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
from mobileposer.evaluate_v2_models import ModulePredictor
from drift_eval_common import angle_between_rotmats

BINS = [(0, 4), (4, 8), (8, 16), (16, 1000)]


def predict(fn, acc, ori, mode):
    T = len(acc)
    if mode == 'full':
        return fn(acc, ori)[:T]
    if mode == 'win125':
        return torch.cat([fn(acc[s:s + 125], ori[s:s + 125])[:min(125, T - s)] for s in range(0, T, 125)])[:T]
    outs = []                                    # win46, every frame costs a pass: evaluate every 3rd frame, hold in between
    last = None
    for t in range(T):
        if t % 3 == 0:
            a, b = max(0, t - 40), min(T, t + 6)
            last = fn(acc[a:b], ori[a:b])[t - a]
        outs.append(last)
    return torch.stack(outs)


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    models = {'v2_base(seen)': ModulePredictor('checkpoints/v2_base_poser', 'checkpoints/v2_base_joints', False, dev),
              'v2_canon(seen)': ModulePredictor('checkpoints/v2_canon_poser', 'checkpoints/v2_canon_joints', True, dev),
              'v3_base(held-out)': ModulePredictor('checkpoints/v3_base_poser', 'checkpoints/v3_base_joints', False, dev),
              'v3_canon(held-out)': ModulePredictor('checkpoints/v3_canon_poser', 'checkpoints/v3_canon_joints', True, dev)}
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
        Rm = pre.mean(0)
        U, S, Vt = torch.linalg.svd(Rm)
        amb = angle_between_rotmats(hold, (U @ Vt).unsqueeze(0).expand(len(hold), 3, 3)).mean().item()
        r = {'amb': amb, 'sub': sub}
        for m, fn in models.items():
            for mode in ('full', 'win125', 'win46'):
                if mode == 'win46' and not m.startswith('v3'):
                    continue
                pred = predict(fn, acc, ori, mode)
                Gp, _ = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
                terr = angle_between_rotmats(Gp.reshape(-1, 24, 3, 3)[:, 1], G[:, 1]).numpy()
                r[(m, mode)] = [terr[t0 + int(lo * 30):min(end, t0 + int(hi * 30))].mean() if min(end, t0 + int(hi * 30)) - (t0 + int(lo * 30)) >= 10 else np.nan
                                for lo, hi in BINS]
        rows.append(r)
        print('  event', sub, i, flush=True)
    amb = np.array([r['amb'] for r in rows])
    keys = [k for k in rows[0] if isinstance(k, tuple)]
    print(f'\n{len(rows)} held-out stand->hold events; ambiguity quartiles {np.round(np.quantile(amb, [0, .25, .5, .75, 1]), 1)}')
    for name, sel in (('ALL', np.ones(len(rows), bool)), ('AMBIGUOUS (<30 deg, standing-like)', amb < 30), ('clearly different (>=30 deg)', amb >= 30)):
        print(f'\n--- {name}: {sel.sum()} events ---   thigh error (deg) by time since flexion onset')
        print('model / inference'.ljust(32) + '  '.join(f'{lo}-{hi if hi < 1000 else "end"}s'.rjust(8) for lo, hi in BINS))
        for k in keys:
            v = np.array([r[k] for r, s in zip(rows, sel) if s])
            print(f'{k[0]} / {k[1]}'.ljust(32) + '  '.join(f'{np.nanmean(v[:, b]):8.1f}' for b in range(len(BINS))))


if __name__ == '__main__':
    main()
