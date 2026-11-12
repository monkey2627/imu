"""Ceiling test for the forearm-anchored upper-arm representation (GT only, no learning).

Given the TRUE upper-arm direction (swing) and the forearm orientation, how well can the upper-arm twist be recovered on
each dataset with priors fitted on AMASS only?
  B  global prior : twist of the upper arm relative to its minimal swing from the rest pose = AMASS mean (no forearm info)
  A  anchored     : choose the twist so that the elbow local rotation's twist about the forearm axis equals its AMASS mean
                    (the forearm orientation, which carries each dataset's calibration convention, fixes the twist)
Reports mean |twist error| and signed mean of the upper-arm (joint 16/17) global twist vs GT.

  python -m mobileposer.stream.twist_ceiling
"""
import numpy as np
import torch
import torch.nn.functional as F

import mobileposer.articulate as art
import mobileposer.evaluate_standard as es
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.evaluate_dip_real import load_dip

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
body = ParametricModel(paths.smpl_file, device=dev)
J = body._J.float().view(24, 3).to(dev)
ARMS = {'L': (16, 18, 20), 'R': (17, 19, 21)}


def rot_about(axis, ang):
    """axis [N,3], ang [N] (rad) -> [N,3,3]"""
    return art.math.axis_angle_to_rotation_matrix((axis * ang.unsqueeze(-1)).reshape(-1, 3)).view(-1, 3, 3)


def min_rot(a, b):
    """minimal rotation taking unit a [3] to unit b [N,3]"""
    a = a.expand_as(b)
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1).clamp(-1, 1)
    ang = torch.acos(c)
    return rot_about(F.normalize(v + 1e-9, dim=-1), ang)


def signed_twist(R, axis):
    """twist angle (rad) of R about unit axis: with q = (w, v), vee(R - R^T) = 4 w v and 4 w^2 = 1 + tr R,
    so 2 atan2(v.a, w) = 2 atan2(vee(R - R^T).a, 1 + tr R)"""
    R = R.reshape(-1, 3, 3)
    vee = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    return 2 * torch.atan2((vee * axis).sum(-1), 1 + R.diagonal(dim1=-2, dim2=-1).sum(-1))


def globals_(poses):
    P = torch.cat(poses).to(dev).clone()
    P[:, 0] = torch.eye(3, device=dev)
    return body.forward_kinematics_R(P).view(-1, 24, 3, 3), P


def quantities(G, P, side):
    s, e, c = ARMS[side]
    au = F.normalize(J[e] - J[s], dim=0); af = F.normalize(J[c] - J[e], dim=0)
    Gu, Gf = G[:, s], G[:, e]
    du = Gu @ au
    S = min_rot(au, du)                                          # swing-only orientation with the true direction
    tw_glob = signed_twist(S.transpose(-1, -2) @ Gu, au)         # upper-arm twist relative to the minimal swing
    tw_elb = signed_twist(P[:, e], af)                           # elbow local twist about the forearm axis
    return dict(au=au, af=af, Gu=Gu, Gf=Gf, du=du, S=S, tw_glob=tw_glob, tw_elb=tw_elb)


def estimate_anchored(q, tw_elb_prior, n_grid=360, chunk=20000):
    """choose psi in G_u = Rot(du, psi) S so that twist(G_u^T G_f, af) is closest to the prior"""
    out = []
    psis = torch.linspace(-np.pi, np.pi, n_grid + 1, device=dev)[:-1]
    for i in range(0, len(q['du']), chunk):
        du, S, Gf = q['du'][i:i + chunk], q['S'][i:i + chunk], q['Gf'][i:i + chunk]
        n = len(du)
        best, best_err = torch.zeros(n, device=dev), torch.full((n,), 1e9, device=dev)
        for psi in psis:
            Gu = rot_about(du, psi.expand(n)) @ S
            t = signed_twist(Gu.transpose(-1, -2) @ Gf, q['af'].expand(n, 3))
            err = torch.atan2(torch.sin(t - tw_elb_prior), torch.cos(t - tw_elb_prior)).abs()
            better = err < best_err
            best[better], best_err[better] = psi, err[better]
        out.append(best)
    psi = torch.cat(out)
    return rot_about(q['du'], psi) @ q['S']


def twist_err(Gu_hat, q):
    t = signed_twist(Gu_hat.transpose(-1, -2) @ q['Gu'], q['au'].expand(len(Gu_hat), 3))
    d = torch.rad2deg(t)
    return d.abs().mean().item(), d.mean().item()


def main():
    ev = es.StdEvaluator(torch.device('cpu'))
    amass = []
    for f in ['CMU', 'BMLmovi', 'KIT', 'ACCAD', 'BioMotionLab_NTroje', 'MPI_HDM05', 'TotalCapture']:
        d = torch.load(f'data/pnp5_60a/pnp5/{f}.pt')
        amass += [p[::8] for p in d['pose'][:200]]
    sets = {'AMASS (prior source)': amass,
            'DIP test GT': [p for _, _, _, p in es.sequences('dipimu', ev.art)],
            'DIP train GT': [s['pose'][::2] for s in load_dip(ParametricModel(paths.smpl_file), subjects=[f's_{i:02d}' for i in range(1, 9)])],
            'TotalCapture GT': [p[::2] for _, _, _, p in es.sequences('totalcapture_dipcalib', ev.art)]}
    G, P = globals_(amass)
    priors = {}
    for side in ARMS:
        q = quantities(G, P, side)
        # circular means of the AMASS twists
        priors[side] = (torch.atan2(torch.sin(q['tw_glob']).mean(), torch.cos(q['tw_glob']).mean()),
                        torch.atan2(torch.sin(q['tw_elb']).mean(), torch.cos(q['tw_elb']).mean()))
        print(f'AMASS prior {side}: global twist {np.rad2deg(priors[side][0].item()):+.1f} deg, elbow twist {np.rad2deg(priors[side][1].item()):+.1f} deg')
    print('\nupper-arm twist error  |mean| (signed mean), deg')
    for name, poses in sets.items():
        G, P = globals_(poses)
        row = []
        for side in ARMS:
            q = quantities(G, P, side)
            n = len(q['du'])
            Gb = rot_about(q['du'], priors[side][0].expand(n)) @ q['S']
            Ga = estimate_anchored(q, priors[side][1])
            b, a = twist_err(Gb, q), twist_err(Ga, q)
            row.append(f'{side}: global prior {b[0]:5.1f} ({b[1]:+5.1f}) | forearm-anchored {a[0]:5.1f} ({a[1]:+5.1f})')
        print(f'  {name:22s} N={len(P):7d}  ' + '   '.join(row), flush=True)


if __name__ == '__main__':
    main()
