"""Is the elbow hinge axis, expressed in the (calibrated) forearm frame, a per-session constant that explains the
per-sequence upper-arm twist error?  GT analysis + correlation with a model's per-sequence signed twist error.

For each sequence and arm (frames with elbow flexion > 30 deg): hinge axis = normal of the upper-arm/forearm plane,
expressed in the FOREARM frame, as an angle around the forearm rest axis.  Reports within-sequence vs between-sequence
spread, and the correlation (over sequences) between the per-sequence hinge angle and the model's signed twist error.

  MOBILEPOSER_FPS=60 python -m mobileposer.stream.hinge_session --model stream_k5_mount8_both_s42
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import mobileposer.evaluate_standard as es
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.stream.diagnose_ft import load, r6d_to_R
from mobileposer.stream.evaluate import predict
from mobileposer.stream.features import REDUCED, targets, twist_angle

ARMS = {'L': (16, 18, 20, 1.0), 'R': (17, 19, 21, -1.0)}


def circ(x):
    return float(np.degrees(np.arctan2(np.sin(x).mean(), np.cos(x).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--datasets', nargs='+', default=['dipimu', 'totalcapture_dipcalib'])
    args = ap.parse_args()
    dev = torch.device('cuda:0')
    body = ParametricModel(paths.smpl_file, device=dev)
    J = body._J.float().view(24, 3).to(dev)
    m, K, g = load(args.model, dev)
    ev = es.StdEvaluator(dev)
    for ds in args.datasets:
        print(f'\n== {ds}')
        rows = {s: [] for s in ARMS}
        for name, a, R, pose in es.sequences(ds, ev.art):
            a, R, pose = a.to(dev), R.to(dev), pose.to(dev)
            Gq = r6d_to_R(targets(body, pose)[1])
            Gp = r6d_to_R(predict(m, a, R, K, g))
            for side, (s, e, c, sg) in ARMS.items():
                fa = F.normalize(J[c] - J[e], dim=0); ua = F.normalize(J[e] - J[s], dim=0)
                ref = F.normalize(torch.cross(fa, torch.tensor([0., 1., 0.], device=dev), dim=0), dim=0); ref2 = torch.cross(fa, ref, dim=0)
                Re = pose[:, e]                                                    # elbow local rotation
                d = Re @ fa
                flex = torch.rad2deg(torch.acos((d @ fa).clamp(-1, 1)))
                msk = flex > 30
                if msk.sum() < 60:
                    continue
                nu = F.normalize(torch.cross(fa.expand_as(d[msk]), d[msk], dim=-1), dim=-1)
                nf = (Re[msk].transpose(-1, -2) @ nu.unsqueeze(-1)).squeeze(-1)    # hinge axis in the forearm frame
                phi = sg * torch.atan2(nf @ ref2, nf @ ref)
                k = REDUCED.index(s)
                tw = twist_angle(Gq[:, k].transpose(-1, -2) @ Gp[:, k], ua)        # signed upper-arm twist error
                rows[side].append((circ(phi.cpu().numpy()), float(np.degrees(np.std(np.angle(np.exp(1j * (phi.cpu().numpy() - np.radians(circ(phi.cpu().numpy())))))))),
                                   circ(tw.cpu().numpy())))
        for side, r in rows.items():
            r = np.array(r)
            if len(r) < 3:
                continue
            corr = np.corrcoef(r[:, 0], r[:, 2])[0, 1]
            print(f'  {side}: n_seq {len(r):2d} | hinge angle per-seq mean {r[:, 0].mean():6.1f}, between-seq sd {r[:, 0].std():5.1f}, '
                  f'mean within-seq sd {r[:, 1].mean():5.1f} | model twist err per-seq mean {r[:, 2].mean():+6.1f} sd {r[:, 2].std():5.1f} | '
                  f'corr(hinge, twist err) {corr:+.2f}')


if __name__ == '__main__':
    main()
