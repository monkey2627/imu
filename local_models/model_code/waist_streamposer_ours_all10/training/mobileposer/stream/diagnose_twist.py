"""Causal check of the fine-tuning damage: take the fine-tuned model's output and transplant the twist (rotation about the
bone axis) of selected joints from the pre-trained model, keeping everything else (swing of those joints, all other
segments' global orientations) from the fine-tuned model.  If TC recovers while DIP keeps most of its gain, the damage
is the label convention of that unobservable twist.

  MOBILEPOSER_FPS=60 python -m mobileposer.stream.diagnose_twist --pre stream_k5_mount8_s42 --post stream_k5_mount8_s42_ftr3
"""
import argparse
import json

import numpy as np
import torch

import mobileposer.articulate as art
import mobileposer.evaluate_standard as es
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.stream.diagnose_ft import load, r6d_to_R, CHILD
from mobileposer.stream.evaluate import predict
from mobileposer.stream.features import REDUCED, to_local_pose

VARIANTS = {'post': [], 'post+twist(shoulders)': [16, 17], 'post+twist(hips)': [1, 2],
            'post+twist(shoulders,hips)': [1, 2, 16, 17], 'post+twist(all)': REDUCED}


def twist_about(R, axis):
    """twist part of R [...,3,3] about unit axis [...,3] (R expressed in the bone frame)"""
    q = art.math.axis_angle_to_quaternion(art.math.rotation_matrix_to_axis_angle(R.reshape(-1, 3, 3))).view(*R.shape[:-2], 4)
    tq = torch.cat([q[..., :1], (q[..., 1:] * axis).sum(-1, keepdim=True) * axis], -1)
    tq = tq / tq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return art.math.quaternion_to_rotation_matrix(tq.reshape(-1, 4)).view(*R.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True); ap.add_argument('--post', required=True)
    args = ap.parse_args()
    dev = torch.device('cuda:0')
    pre, post = load(args.pre, dev), load(args.post, dev)
    body = ParametricModel(paths.smpl_file, device=dev)
    J = body._J.to(dev).float().view(24, 3)
    axes = torch.stack([torch.nn.functional.normalize(J[CHILD[j]] - J[j], dim=0) if CHILD[j] != j else torch.tensor([0., 1., 0.], device=dev) for j in REDUCED])
    ev = es.StdEvaluator(dev)
    res = {}
    for ds in ('dipimu', 'totalcapture_dipcalib', 'totalcapture_officalib'):
        errs = {v: [] for v in ['pre'] + list(VARIANTS)}
        for name, a, R, pose in es.sequences(ds, ev.art):
            a, R = a.to(dev), R.to(dev)
            G0, G1 = r6d_to_R(predict(*[pre[0], a, R, pre[1], pre[2]])), r6d_to_R(predict(*[post[0], a, R, post[1], post[2]]))
            errs['pre'].append(ev(to_local_pose(body, art.math.rotation_matrix_to_r6d(G0.reshape(-1, 3, 3)).view(len(a), -1), R[:, 4]).cpu(), pose))
            for v, joints in VARIANTS.items():
                G = G1.clone()
                for j in joints:
                    k = REDUCED.index(j)
                    D = G1[:, k].transpose(-1, -2) @ G0[:, k]                          # post -> pre, in the post bone frame
                    G[:, k] = G1[:, k] @ twist_about(D, axes[k].expand(len(D), 3))
                r6d = art.math.rotation_matrix_to_r6d(G.reshape(-1, 3, 3)).view(len(a), -1)
                errs[v].append(ev(to_local_pose(body, r6d, R[:, 4]).cpu(), pose))
        res[ds] = {v: torch.stack(e).mean(0).tolist() for v, e in errs.items()}
        print(f'\n{ds}  SIP/Ang/Pos/Mesh')
        for v, m in res[ds].items():
            print('  %-28s %s' % (v, np.round(m, 2)), flush=True)
    json.dump(res, open(f'results/stream/diagnose_twist_{args.post}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
