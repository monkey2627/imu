"""Why does DIP fine-tuning hurt TotalCapture?  Compares a pre-trained and a fine-tuned streaming model (inference only).

Per dataset (DIP test, TC DIP-calib, TC official, AMASS held-out) and per reduced joint, the root-relative global error
rotation E = G_gt^T G_pred is split into
  - per-sequence constant bias (chordal mean of E over the sequence) vs the residual around it   -> convention shift (H1)
  - swing vs twist about the bone axis                                                              -> upper-arm twist etc.
Plus: robustness to injected constant sensor-mount offsets on AMASS held-out (H2), per-action deltas on TC (H3).

  MOBILEPOSER_FPS=60 python -m mobileposer.stream.diagnose_ft --pre stream_k5_mount8_s42 --post stream_k5_mount8_s42_ftr3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import mobileposer.evaluate_standard as es
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.stream.data import Store
from mobileposer.stream.evaluate import predict
from mobileposer.stream.features import REDUCED, targets, random_mount
from mobileposer.stream.model import StreamPoser
import mobileposer.articulate as art

CK = Path('checkpoints_research')
CHILD = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 9: 12, 12: 15, 13: 16, 14: 17, 15: 15, 16: 18, 17: 19, 18: 20, 19: 21}
NAME = {1: 'Lhip', 2: 'Rhip', 3: 'spine1', 4: 'Lknee', 5: 'Rknee', 6: 'spine2', 9: 'spine3', 12: 'neck', 13: 'Lcollar',
        14: 'Rcollar', 15: 'head', 16: 'Lshoulder', 17: 'Rshoulder', 18: 'Lelbow', 19: 'Relbow'}
SHOW = [1, 2, 4, 5, 9, 13, 14, 16, 17, 18, 19]


def load(name, dev):
    cfg = json.load(open(CK / name / 'config.json'))
    m = StreamPoser(bool(cfg['gyro']), cfg['hidden'], cfg['layers'], anchor=cfg.get('anchor', 'none'), frame=cfg.get('frame', 'rel'), arch=cfg.get('arch', 'mp2'),
                    local=bool(cfg.get('local_gyro', 0))).to(dev).eval()
    m.load_state_dict(torch.load(CK / name / 'model.pt', map_location=dev))
    return m, cfg['lag'], bool(cfg['gyro'])


def r6d_to_R(r):
    return art.math.r6d_to_rotation_matrix(r.reshape(-1, 6)).view(*r.shape[:-1], len(REDUCED), 3, 3) if r.shape[-1] == 90 else None


def angle(R):
    return torch.acos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)) * 180 / np.pi


def chordal_mean(R):
    U, _, Vt = torch.linalg.svd(R.mean(0))
    D = torch.eye(3, device=R.device).repeat(*U.shape[:-2], 1, 1)
    D[..., 2, 2] = torch.det(U @ Vt)
    return U @ D @ Vt


def swing_twist(E, axis):
    """E [...,3,3] in the bone frame, axis [...,3] unit -> twist, swing angles (deg)"""
    q = art.math.axis_angle_to_quaternion(art.math.rotation_matrix_to_axis_angle(E.reshape(-1, 3, 3))).view(*E.shape[:-2], 4)
    proj = (q[..., 1:] * axis).sum(-1)
    twist = 2 * torch.atan2(proj.abs(), q[..., 0].abs()) * 180 / np.pi
    tw_q = torch.cat([q[..., :1], proj.unsqueeze(-1) * axis], -1)
    tw_q = tw_q / tw_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # swing = q * twist^-1 ; its angle = 2 acos |<q, twist>|
    swing = 2 * torch.acos((q * tw_q).sum(-1).abs().clamp(max=1)) * 180 / np.pi
    return twist, swing


def analyse(seqs, models, axes, dev):
    """seqs: list of (name, a, R, r6d_gt).  Returns per model: dict joint -> stats, and per-sequence SIP-joint mean."""
    out = {}
    for tag, (m, K, gyro) in models.items():
        acc = {j: {'total': [], 'bias': [], 'resid': [], 'twist': [], 'swing': [], 'bias_twist': [], 'bias_swing': []} for j in REDUCED}
        per_seq = {}
        for name, a, R, gt in seqs:
            p = predict(m, a, R, K, gyro)
            Gp, Gq = r6d_to_R(p), r6d_to_R(gt)
            E = Gq.transpose(-1, -2) @ Gp                                                     # [T,15,3,3] bone frame
            tot = angle(E)
            B = chordal_mean(E)                                                                # [15,3,3]
            res = angle(B.transpose(-1, -2).unsqueeze(0) @ E)
            tw, sw = swing_twist(E, axes)
            btw, bsw = swing_twist(B, axes)
            for k, j in enumerate(REDUCED):
                for key, v in (('total', tot[:, k].mean()), ('bias', angle(B[k])), ('resid', res[:, k].mean()),
                               ('twist', tw[:, k].mean()), ('swing', sw[:, k].mean()), ('bias_twist', btw[k]), ('bias_swing', bsw[k])):
                    acc[j][key].append(float(v))
            per_seq[name] = float(tot[:, [REDUCED.index(j) for j in (1, 2, 16, 17)]].mean())
        out[tag] = ({NAME[j]: {k: float(np.mean(v)) for k, v in acc[j].items()} for j in REDUCED}, per_seq)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True); ap.add_argument('--post', required=True)
    ap.add_argument('--max_val', type=int, default=120)
    args = ap.parse_args()
    dev = torch.device('cuda:0')
    models = {'pre': load(args.pre, dev), 'post': load(args.post, dev)}
    body = ParametricModel(paths.smpl_file, device=dev)
    J = body._J.to(dev).float() if hasattr(body, '_J') else body.get_zero_pose_joint_and_vertex()[0].to(dev)
    J = J.view(24, 3)
    axes = torch.stack([torch.nn.functional.normalize(J[CHILD[j]] - J[j], dim=0) if CHILD[j] != j else torch.tensor([0., 1., 0.], device=dev) for j in REDUCED])
    ev = es.StdEvaluator(dev)
    report = {}

    def test_seqs(ds):
        return [(str(n), a.to(dev), R.to(dev), targets(body, p.to(dev))[1]) for n, a, R, p in es.sequences(ds, ev.art)]

    val = Store('data/stream_pnp5_60a_val.pt', dev)
    vseqs = []
    for i in range(min(args.max_val, len(val))):
        o, n = int(val.offs[i]), int(val.lens[i])
        vseqs.append((val.names[i], val.acc[o:o + n].view(n, 5, 3), val.ori[o:o + n].view(n, 5, 3, 3), val.r6d[o:o + n]))

    for ds, seqs in (('dipimu', None), ('totalcapture_dipcalib', None), ('totalcapture_officalib', None), ('amass_val', vseqs)):
        seqs = seqs if seqs is not None else test_seqs(ds)
        res = analyse(seqs, models, axes, dev)
        report[ds] = {k: v[0] for k, v in res.items()}
        report[ds + '_perseq'] = {k: v[1] for k, v in res.items()}
        print(f'\n== {ds}   (deg; total = mean error, bias = per-sequence constant part, resid = around the bias)')
        print('  %-10s %-5s %6s %6s %6s %6s %6s %8s %8s' % ('joint', 'model', 'total', 'bias', 'resid', 'twist', 'swing', 'b_twist', 'b_swing'))
        for j in SHOW:
            for tag in ('pre', 'post'):
                s = report[ds][tag][NAME[j]]
                print('  %-10s %-5s %6.1f %6.1f %6.1f %6.1f %6.1f %8.1f %8.1f' % (NAME[j], tag, s['total'], s['bias'], s['resid'], s['twist'], s['swing'], s['bias_twist'], s['bias_swing']))
        if ds.startswith('totalcapture'):
            import re, collections
            names = [str(n) for n, *_ in es.sequences('totalcapture_officalib', ev.art)]
            g = collections.defaultdict(list)
            for (k, pre), post, nm in zip(report[ds + '_perseq']['pre'].items(), report[ds + '_perseq']['post'].values(), names):
                g[re.sub(r'\d+$', '', nm.split('_', 1)[1])].append(post - pre)
            print('  SIP-joint delta (post - pre) by action:', ' '.join(f'{k} {np.mean(v):+.2f}' for k, v in sorted(g.items())))

    # H2: robustness to constant sensor-mount offsets (AMASS held-out, fixed random offsets per sequence)
    print('\n== mount-offset robustness on AMASS held-out: SIP-joint error (deg)')
    report['mount'] = {}
    for sigma in (0, 5, 10, 15):
        row = {}
        for tag, (m, K, gyro) in models.items():
            errs = []
            torch.manual_seed(1)
            for name, a, R, gt in vseqs:
                M = random_mount(1, sigma, dev)[0] if sigma else torch.eye(3, device=dev).expand(5, 3, 3)
                p = predict(m, a, R @ M, K, gyro)
                e = angle(r6d_to_R(gt).transpose(-1, -2) @ r6d_to_R(p))
                errs.append(float(e[:, [REDUCED.index(j) for j in (1, 2, 16, 17)]].mean()))
            row[tag] = float(np.mean(errs))
        report['mount'][sigma] = row
        print('  sigma %2d deg: pre %.2f  post %.2f  (post - pre %+.2f)' % (sigma, row['pre'], row['post'], row['post'] - row['pre']))
    Path('results/stream').mkdir(parents=True, exist_ok=True)
    json.dump(report, open(f'results/stream/diagnose_ft_{args.post}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
