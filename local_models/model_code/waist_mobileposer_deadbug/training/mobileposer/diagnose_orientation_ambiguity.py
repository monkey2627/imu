"""How much of the thigh orientation / knee flexion is determined by the WORLD orientations
of the waist (spine1) and shank sensors alone?  (no training, no network)

Non-parametric estimate from AMASS: for a query frame, find frames from OTHER sequences whose
waist AND shank global orientations are both within ~10 deg of the query's; then
  - 'thigh error'   = geodesic angle between the neighbour's thigh orientation and the query's true thigh orientation
  - 'knee flex err' = |knee flexion(neighbour) - knee flexion(query)|
A neighbour-based estimator that only sees (waist, shank) orientations can do no better than the
spread among such neighbours. Large spread => orientation-only sensing leaves the limb ambiguous
(e.g. sitting vs standing both have an upright waist and an upright shank).

Part 1: AMASS queries, split by true knee flexion.
Part 2: the 10 real captures' reference frames used as queries (sitting / standing / knee bend ...).

Run from SparseIMU/:  python -m mobileposer.diagnose_orientation_ambiguity
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.orientation_fusion import _so3_log
from drift_eval_common import angle_between_rotmats

SUBSETS = ['CMU', 'Transitions_mocap', 'BMLmovi', 'ACCAD', 'DanceDB', 'Eyes_Japan_Dataset', 'BMLhandball']
KNEE_AXIS = torch.tensor([0.99, -0.01, -0.15])
RADIUS = 0.35          # ~ 10 deg on each of the two orientations (Frobenius distance of stacked 3x3 matrices)


def knee_flex_deg(local_knee):
    v = _so3_log(local_knee)
    return torch.rad2deg((v * KNEE_AXIS / KNEE_AXIS.norm()).sum(-1))


def main():
    body = ParametricModel(paths.smpl_file)
    root = Path('data/no_head_5imu_surface_processed/wrists_shanks_waist')
    Gw, Gs, Gt, flex, seq = [], [], [], [], []
    sid = 0
    for sub in SUBSETS:
        d = torch.load(root / f'{sub}.pt', map_location='cpu', weights_only=False)
        for p in d['pose'][:600]:
            p = p[::6]
            if len(p) < 5:
                continue
            G, _ = body.forward_kinematics(p.reshape(-1, 216), torch.ones(10), tran=None, calc_mesh=False)
            G = G.reshape(-1, 24, 3, 3)
            Gw.append(G[:, 3]); Gs.append(G[:, 4]); Gt.append(G[:, 1])
            flex.append(knee_flex_deg(p.reshape(-1, 24, 3, 3)[:, 4]))
            seq.append(torch.full((len(p),), sid)); sid += 1
        del d
    Gw, Gs, Gt, flex, seq = torch.cat(Gw), torch.cat(Gs), torch.cat(Gt), torch.cat(flex), torch.cat(seq).numpy()
    N = len(Gw)
    print(f'database frames {N} from {sid} sequences (every 6th frame; left leg)')
    feats = torch.cat([Gw.reshape(N, 9), Gs.reshape(N, 9)], 1).numpy()
    tree = cKDTree(feats)

    def query(Qw, Qs, Qt, Qflex, qseq=None, max_q=4000, rng=np.random.RandomState(0)):
        idx = rng.permutation(len(Qw))[:max_q]
        out = []
        for i in idx:
            nb = tree.query_ball_point(torch.cat([Qw[i].reshape(9), Qs[i].reshape(9)]).numpy(), RADIUS)
            if qseq is not None:
                nb = [j for j in nb if seq[j] != qseq[i]]
            if len(nb) < 5 or len(set(seq[nb])) < 2:
                out.append((i, len(nb), np.nan, np.nan, np.nan, np.nan))
                continue
            nb = np.array(nb)
            t_err = angle_between_rotmats(Gt[nb], Qt[i].unsqueeze(0).expand(len(nb), 3, 3)).numpy()
            f_err = np.abs(flex[nb].numpy() - float(Qflex[i]))
            out.append((i, len(nb), float(np.mean(t_err)), float(np.median(t_err)), float(np.mean(f_err)), float(flex[nb].std())))
        return np.array(out, dtype=float)

    print('\n=== Part 1: AMASS queries (neighbours from OTHER sequences) ===')
    res = query(Gw, Gs, Gt, flex, qseq=seq)
    bins = [('all', lambda f: np.ones_like(f, bool)), ('knee flex < 20 deg', lambda f: f < 20),
            ('20-60 deg', lambda f: (f >= 20) & (f < 60)), ('>= 60 deg (deep flexion)', lambda f: f >= 60)]
    print(f'{"query set":28s} n_q  covered  mean thigh err  median  mean knee-flex err  neighbour flex std')
    for name, fn in bins:
        sel = fn(flex.numpy()[res[:, 0].astype(int)])
        r = res[sel]
        ok = r[~np.isnan(r[:, 2])]
        if len(r) == 0:
            continue
        print(f'{name:28s} {len(r):4d}  {len(ok) / len(r) * 100:5.0f}%   {np.mean(ok[:, 2]) if len(ok) else float("nan"):10.1f}   '
              f'{np.mean(ok[:, 3]) if len(ok) else float("nan"):6.1f}   {np.mean(ok[:, 4]) if len(ok) else float("nan"):12.1f}        '
              f'{np.mean(ok[:, 5]) if len(ok) else float("nan"):8.1f}')

    print('\n=== Part 2: the 10 real captures (reference frames as queries, left leg) ===')
    print(f'{"exercise":16s} true knee flex(mean)  frames covered  neighbour-implied knee flex (mean)  thigh err vs truth (mean)')
    cap_root = Path('results/ours_multi_finetune/captures')
    for cap in sorted(cap_root.iterdir(), key=lambda p: int(p.name.split('_')[0])):
        t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        p = t['pose'].reshape(-1, 24, 3, 3)[::6]
        G, _ = body.forward_kinematics(p.reshape(-1, 216), torch.ones(10), tran=None, calc_mesh=False)
        G = G.reshape(-1, 24, 3, 3)
        fl = knee_flex_deg(p[:, 4])
        implied, terr, cov = [], [], 0
        for i in range(len(p)):
            nb = tree.query_ball_point(torch.cat([G[i, 3].reshape(9), G[i, 4].reshape(9)]).numpy(), RADIUS)
            if len(nb) < 5 or len(set(seq[nb])) < 2:
                continue
            nb = np.array(nb); cov += 1
            implied.append(float(flex[nb].mean()))
            terr.append(float(angle_between_rotmats(Gt[nb], G[i, 1].unsqueeze(0).expand(len(nb), 3, 3)).mean()))
        print(f'{cap.name.split("__")[0]:16s} {float(fl.mean()):12.1f}        {cov / len(p) * 100:5.0f}%        '
              f'{np.mean(implied) if implied else float("nan"):10.1f}                     {np.mean(terr) if terr else float("nan"):8.1f}')


if __name__ == '__main__':
    main()
