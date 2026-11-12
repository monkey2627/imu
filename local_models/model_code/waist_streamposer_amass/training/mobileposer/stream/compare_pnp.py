"""Where does PNP get its DIP shoulder/hip accuracy without losing TotalCapture?  Same error decomposition as
diagnose_ft.py for PNP (dumped by scripts_run/pnp_dump.py) and our pre/post fine-tune models, plus the SIGNED twist error
(mean over frames, then over sequences): a convention offset shows up as a consistent sign.

  MOBILEPOSER_FPS=60 python -m mobileposer.stream.compare_pnp --ours stream_k5_mount8_s42 stream_k5_mount8_s42_ftr3
"""
import argparse

import numpy as np
import torch

import mobileposer.articulate as art
import mobileposer.evaluate_standard as es
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.stream.diagnose_ft import load, r6d_to_R, angle, chordal_mean, swing_twist, CHILD
from mobileposer.stream.evaluate import predict
from mobileposer.stream.features import REDUCED, targets

JOINTS = {1: 'Lhip', 2: 'Rhip', 16: 'Lshoulder', 17: 'Rshoulder', 18: 'Lelbow', 19: 'Relbow'}


def signed_twist(E, axis):
    q = art.math.axis_angle_to_quaternion(art.math.rotation_matrix_to_axis_angle(E.reshape(-1, 3, 3))).view(*E.shape[:-2], 4)
    s = (q[..., 1:] * axis).sum(-1)
    return 2 * torch.atan2(s * torch.sign(q[..., 0]), q[..., 0].abs()) * 180 / np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', nargs='+', required=True)
    ap.add_argument('--pnp', nargs='+', default=['tpose'])
    ap.add_argument('--datasets', nargs='+', default=['dipimu', 'totalcapture_dipcalib', 'totalcapture_officalib'])
    ap.add_argument('--skip_s', type=float, default=0.0, help='ignore the first seconds of every sequence')
    args = ap.parse_args()
    dev = torch.device('cuda:0')
    body = ParametricModel(paths.smpl_file, device=dev)
    J = body._J.to(dev).float().view(24, 3)
    axes = torch.stack([torch.nn.functional.normalize(J[CHILD[j]] - J[j], dim=0) if CHILD[j] != j else torch.tensor([0., 1., 0.], device=dev) for j in REDUCED])
    ev = es.StdEvaluator(dev)
    models = {n: load(n, dev) for n in args.ours}
    for ds in args.datasets:
        if ds.startswith('synth_'):   # motion sets with clean synthetic IMU (scripts_run/make_synth_sets.py + pnp_synth.py)
            syn = np.load(f'results/stream/{ds}_inputs.npz')
            n = len([k for k in syn.files if k.startswith('pose')])
            seqs = [(torch.from_numpy(syn[f'a{i}']).to(dev), torch.from_numpy(syn[f'R{i}']).to(dev),
                     r6d_to_R(targets(body, torch.from_numpy(syn[f'pose{i}']).to(dev))[1])) for i in range(n)]
        elif ds == 'dipsynth':        # DIP test GT motions with clean synthetic IMU (scripts_run/pnp_synth_dip.py)
            syn = np.load('results/stream/dipsynth_inputs.npz')
            seqs = [(torch.from_numpy(syn[f'a{i}']).to(dev), torch.from_numpy(syn[f'R{i}']).to(dev), r6d_to_R(targets(body, p.to(dev))[1]))
                    for i, (_, _, _, p) in enumerate(es.sequences('dipimu', ev.art))]
        else:
            seqs = [(a.to(dev), R.to(dev), r6d_to_R(targets(body, p.to(dev))[1])) for _, a, R, p in es.sequences(ds, ev.art)]
        preds = {n: [r6d_to_R(predict(m, a, R, K, g)) for a, R, _ in seqs] for n, (m, K, g) in models.items()}
        for init in args.pnp:
            d = np.load(f'results/stream/dump_pnp_{init}_{ds}.npz')
            preds[f'PNP_{init}'] = [torch.from_numpy(d[str(i)]).to(dev) for i in range(len(seqs))]
        print(f'\n== {ds}: per joint  total | bias(|.|) | signed twist err (mean, sd over seqs) | bias twist | bias swing | resid')
        for j, jn in JOINTS.items():
            k = REDUCED.index(j)
            for n, P in preds.items():
                tot, bias, st, btw, bsw, res = [], [], [], [], [], []
                for (a, R, Gq), Gp in zip(seqs, P):
                    T = min(len(Gq), len(Gp))
                    t0 = int(args.skip_s * 60)                      # drop the cold-start transient
                    if T - t0 < 60:
                        continue
                    E = Gq[t0:T, k].transpose(-1, -2) @ Gp[t0:T, k]
                    T = T - t0
                    B = chordal_mean(E)
                    tw, sw = swing_twist(B, axes[k])
                    tot.append(float(angle(E).mean())); bias.append(float(angle(B))); btw.append(float(tw)); bsw.append(float(sw))
                    st.append(float(signed_twist(E, axes[k].expand(T, 3)).mean()))
                    res.append(float(angle(B.transpose(-1, -2) @ E).mean()))
                print('  %-9s %-28s %5.1f | %5.1f | %+6.1f (%4.1f) | %5.1f | %5.1f | %5.1f' % (
                    jn, n, np.mean(tot), np.mean(bias), np.mean(st), np.std(st), np.mean(btw), np.mean(bsw), np.mean(res)), flush=True)


if __name__ == '__main__':
    main()
