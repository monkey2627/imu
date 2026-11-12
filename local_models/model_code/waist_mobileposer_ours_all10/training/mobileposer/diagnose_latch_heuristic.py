"""Training-free upper-bound test of a posture LATCH on the unsensed bones.

Observation (DIP s_09/05, a 75 s sitting sequence): right after sitting down the causal model is right (thigh
direction error 21 deg) but within ~5 s its estimate decays towards the 'standing' prior and the error grows to
55-80 deg while the person sits still for a minute; offline/windowed baselines are just as wrong (30-70 deg).

Latch: the thigh is expressed relative to its SENSED neighbour (the shank): K = R_shank^T R_thigh (the knee
state, one scalar of flexion plus small residuals).  K is refreshed from the network only while the body is
moving (evidence is informative) and HELD while it is quasi-static:
    moving_t  = max over the last W frames of the angle change of the waist->shank relative rotation (deg per 5 frames) > tau
    K_t       = K_inst,t if moving_t else K_{t-1}
    thigh_t   = R_shank,t K_t            (so the thigh keeps following the shank while the knee state is held)
Only the left/right thigh (joints 1, 2 w.r.t. shanks 4, 5) is latched.  Gate from the measured orientations only (causal).

Reported: thigh orientation / direction error (deg): all DIP test sequences, the sitting sequence, and the 68 held-out
AMASS events (full-sequence causal inference), for the instantaneous source with and without the latch.

Run from SparseIMU/:  python -m mobileposer.diagnose_latch_heuristic
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
from mobileposer.evaluate_causal_holds import load
from mobileposer.evaluate_dip_real import load_dip
from mobileposer.evaluate_v2_models import ModulePredictor
from mobileposer.evaluate_causal_holds import BINS
from drift_eval_common import angle_between_rotmats

TAUS = [2.0, 4.0, 8.0]
W = 15
THIGH = [(1, 4), (2, 5)]          # (thigh joint, shank joint); sensed orientation slots: shank L=2, shank R=3, waist=4
SLOT = {4: 2, 5: 3}


def gate(ori, tau):
    """moving flag per frame (causal): relative waist->shank rotation changed by > tau deg within the last 5 frames, any time in the last W frames"""
    flags = []
    for sj, slot in SLOT.items():
        rel = ori[:, 4].transpose(-1, -2) @ ori[:, slot]
        ch = torch.zeros(len(rel))
        ch[5:] = angle_between_rotmats(rel[5:], rel[:-5])
        flags.append(ch)
    ch = torch.stack(flags).max(0).values
    mv = (ch > tau).float()
    # causal max over the last W frames
    pad = torch.cat([torch.zeros(W - 1), mv])
    return (pad.unfold(0, W, 1).max(-1).values > 0)


def apply_latch(Gp, ori, moving, thigh_idx, shank_idx):
    """Gp [T,24,3,3] predicted global rotations -> latched thigh global rotation [T,3,3]"""
    Rs, Rt = Gp[:, shank_idx], Gp[:, thigh_idx]
    K_inst = Rs.transpose(-1, -2) @ Rt
    K = K_inst.clone()
    for t in range(1, len(K)):
        if not moving[t]:
            K[t] = K[t - 1]
    return Rs @ K


def thigh_errors(body, G, Gp_thigh, J, rest, which):
    """orientation error and rest-bone direction error for the thigh of side `which` (0/1)"""
    ti = THIGH[which][0]
    e = angle_between_rotmats(Gp_thigh, G[:, ti])
    v_p, v_t = (Gp_thigh @ rest[which].view(3, 1)).squeeze(-1), (G[:, ti] @ rest[which].view(3, 1)).squeeze(-1)
    d = torch.rad2deg(torch.acos(((v_p * v_t).sum(-1) / (v_p.norm(dim=-1) * v_t.norm(dim=-1))).clamp(-1, 1)))
    return e, d


def evaluate(body, fn_src, acc, ori, pose, taus, t_slice=None):
    T = len(pose)
    G, J = body.forward_kinematics(pose, torch.ones(10), tran=None, calc_mesh=False)
    G, J = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
    rest = [(G[:, ti].transpose(-1, -2) @ (J[:, ti + 3] - J[:, ti]).unsqueeze(-1)).squeeze(-1).mean(0) for ti, _ in THIGH]   # thigh bone vector in thigh frame
    Gp, _ = body.forward_kinematics(fn_src(acc, ori)[:T], torch.ones(10), tran=None, calc_mesh=False)
    Gp = Gp.reshape(T, 24, 3, 3)
    sl = t_slice if t_slice is not None else slice(0, T)
    out = {}
    cfgs = [('none', None)] + [(f'latch tau={tau}', tau) for tau in taus]
    for name, tau in cfgs:
        es, ds = [], []
        for w, (ti, si) in enumerate(THIGH):
            if tau is None:
                th = Gp[:, ti]
            else:
                th = apply_latch(Gp, ori, gate(ori, tau), ti, si)
            e, d = thigh_errors(body, G, th, J, rest, w)
            es.append(e[sl]); ds.append(d[sl])
        out[name] = (torch.stack(es).mean().item(), torch.stack(ds).mean().item(), torch.stack(ds).mean(0).numpy() if False else None)
    return out


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    sources = {'causal125': load('checkpoints/v3_causal125_poser', 'checkpoints/v3_causal125_joints', True, dev),
               'base offline': ModulePredictor('checkpoints/v3_base_poser', 'checkpoints/v3_base_joints', False, dev)}
    names = ['none'] + [f'latch tau={t}' for t in TAUS]
    print('=== DIP-IMU test (real IMU): thigh orientation/direction error (deg), frame-weighted ===')
    seqs = load_dip(body)
    for src, fn in sources.items():
        agg = {n: [] for n in names}; sit = None; wts = []
        for s in seqs:
            r = evaluate(body, fn, s['acc'], s['ori'], s['pose'], TAUS)
            wts.append(len(s['pose']))
            for n in names:
                agg[n].append(r[n][:2])
            if s['name'] == 's_09/05.pkl':
                sit = r
        print(f'\n[{src}]'.ljust(24) + 'all sequences            sitting sequence s_09/05')
        for n in names:
            a = np.average(agg[n], axis=0, weights=wts)
            print(f'  {n:16s}{a[0]:8.1f}/{a[1]:5.1f}              {sit[n][0]:8.1f}/{sit[n][1]:5.1f}')
    print('\n=== 68 held-out AMASS stand->hold events: thigh error (deg) by time since flexion onset ===')
    blob = torch.load('data/hold_events_heldout.pt', map_location='cpu', weights_only=False)
    for src, fn in sources.items():
        rows = {n: [] for n in names}
        for sub, i, t0, end, T in blob['events']:
            h = blob['held'][sub]
            j = h['indices'].index(i)
            acc, ori, pose = h['acc'][j][:3000], h['ori'][j][:3000], h['pose'][j].reshape(-1, 24, 3, 3)[:3000]
            T = len(pose)
            end = min(end, T)
            for n, tau in [('none', None)] + [(f'latch tau={t}', t) for t in TAUS]:
                per = []
                for lo, hi in BINS:
                    a, b = t0 + int(lo * 30), min(end, t0 + int(hi * 30))
                    per.append(np.nan if b - a < 10 else evaluate_slice(body, fn, acc, ori, pose, tau, slice(a, b)))
                rows[n].append(per)
        print(f'\n[{src}]'.ljust(24) + '  '.join(f'{lo}-{hi if hi < 1000 else "end"}s'.rjust(8) for lo, hi in BINS))
        for n in names:
            print(f'  {n:16s}' + '  '.join(f'{x:8.1f}' for x in np.nanmean(np.array(rows[n]), axis=0)))


_cache = {}


def evaluate_slice(body, fn, acc, ori, pose, tau, sl):
    key = (id(fn), id(pose))
    if key not in _cache:
        T = len(pose)
        G, J = body.forward_kinematics(pose, torch.ones(10), tran=None, calc_mesh=False)
        G, J = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
        Gp, _ = body.forward_kinematics(fn(acc, ori)[:T], torch.ones(10), tran=None, calc_mesh=False)
        _cache.clear()
        _cache[key] = (G, J, Gp.reshape(T, 24, 3, 3))
    G, J, Gp = _cache[key]
    rest = [(G[:, ti].transpose(-1, -2) @ (J[:, ti + 3] - J[:, ti]).unsqueeze(-1)).squeeze(-1).mean(0) for ti, _ in THIGH]
    es = []
    for w, (ti, si) in enumerate(THIGH):
        th = Gp[:, ti] if tau is None else apply_latch(Gp, ori, gate(ori, tau), ti, si)
        e, _ = thigh_errors(body, G, th, J, rest, w)
        es.append(e[sl])
    return torch.stack(es).mean().item()


if __name__ == '__main__':
    main()
