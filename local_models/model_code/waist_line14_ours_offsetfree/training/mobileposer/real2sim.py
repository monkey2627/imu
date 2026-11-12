"""Historical Real2Sim augmentation for PNP-aligned synthetic IMU data.

The retained experiment models high-frequency acceleration and sensor
orientation noise from DIP training subjects. It does not estimate or change
sensor attachment locations. Any use must start from the fixed PNP5 cache and
is an auxiliary data augmentation, not the line14 primary innovation.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_dip_real import load_dip

FPS = int(os.environ.get('MOBILEPOSER_FPS', 30))
SMOOTH = 4                                   # half-width (frames) of the synthesis smoothing (no_head_layouts._syn_acc, smooth_n = 4)
TRAIN = [f's_{i:02d}' for i in range(1, 8)]
VAL = ['s_08']


# ----------------------------------------------------------------------------- signal helpers
def tri_smooth(x, n=SMOOTH):
    """triangular smoothing along dim 0 (kernel (n-|k|)/n^2 == second difference with spacing n applied to the position)"""
    k = torch.tensor([(n - abs(i)) / n ** 2 for i in range(-n, n + 1)], dtype=x.dtype, device=x.device)
    sh = x.shape
    y = x.reshape(sh[0], -1).T.unsqueeze(1)                                   # [C,1,T]
    y = torch.nn.functional.pad(y, (n, n), mode='replicate')
    return torch.nn.functional.conv1d(y, k.view(1, 1, -1)).squeeze(1).T.reshape(sh)


def second_diff(x, n=SMOOTH, fps=FPS):
    """(x[t+n] + x[t-n] - 2 x[t]) * fps^2 / n^2 along dim 0 (replicate padding)"""
    p = torch.cat([x[:1].expand(n, *x.shape[1:]), x, x[-1:].expand(n, *x.shape[1:])])
    return (p[2 * n:] + p[:-2 * n] - 2 * p[n:-n]) * fps * fps / (n * n)


def sequences(body, subjects):
    out = []
    for s in load_dip(body, subjects=subjects):
        T = len(s['pose'])
        G, J = body.forward_kinematics(s['pose'], torch.zeros(10), tran=None, calc_mesh=False)
        s['G'], s['J'] = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
        out.append(s)
    return out


# ----------------------------------------------------------------------------- high-band simulator
def rot6d(R):
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def features(G, J, fps=FPS):
    """reference-motion features [T,F]: joint velocities and accelerations (root-relative, world frame), global rotations (6D)"""
    Jr = J - J[:, :1]
    vel = (torch.cat([Jr[1:], Jr[-1:]]) - torch.cat([Jr[:1], Jr[:-1]])) * fps / 2
    acc = tri_smooth(second_diff(Jr, n=2, fps=fps), 2) / 10
    return torch.cat([vel.flatten(1), acc.flatten(1), rot6d(G).flatten(1)], dim=-1)


class HFSim(nn.Module):
    def __init__(self, n_feat=72 + 72 + 144, hidden=128, n_sensor=5):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * hidden, n_sensor * 3 * 2)
        self.n = n_sensor

    def forward(self, f):
        h, _ = self.gru(self.inp(f))
        mu, lv = self.out(h).chunk(2, dim=-1)
        return mu.reshape(*mu.shape[:-1], self.n, 3), lv.clamp(-8, 4).reshape(*lv.shape[:-1], self.n, 3)


def yaw_rot(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.zeros(*theta.shape, 3, 3, device=theta.device)
    R[..., 0, 0], R[..., 0, 2], R[..., 1, 1], R[..., 2, 0], R[..., 2, 2] = c, s, 1.0, -s, c
    return R


def windows(seqs, W=250, stride=125):
    out = []
    for s in seqs:
        G, J = s['G'], s['J']
        hf = (s['acc'] - tri_smooth(s['acc'])) / 10                              # real high band, [T,5,3]
        T = len(G)
        for a in range(0, T - W + 1, stride):
            out.append((G[a:a + W], J[a:a + W], hf[a:a + W]))
    return out


def pack(items):
    """features and targets of all windows, computed once"""
    return torch.stack([features(g, j) for g, j, _ in items]), torch.stack([h for _, _, h in items])


def yaw_aug(f, h):
    """heading invariance on precomputed tensors: every feature block is a list of 3-vectors, rotate them (and the target) about the vertical axis"""
    B, W = f.shape[:2]
    Ry = yaw_rot(torch.rand(B, device=f.device) * 2 * np.pi)
    f = (f.reshape(B, W, -1, 3) @ Ry.transpose(-1, -2).unsqueeze(1)).reshape(B, W, -1)
    h = h @ Ry.transpose(-1, -2).unsqueeze(1)
    return f, h


def cmd_train(args):
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    tr, va = windows(sequences(body, TRAIN)), windows(sequences(body, VAL))
    print('windows: train', len(tr), 'val', len(va), flush=True)
    net = HFSim().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bank = 1e9, None
    Ftr, Htr = (t.to(dev) for t in pack(tr))
    Fva, Hva = (t.to(dev) for t in pack(va))
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Ftr), device=dev)
        for i in range(0, len(Ftr), 16):
            f, h = yaw_aug(Ftr[perm[i:i + 16]], Htr[perm[i:i + 16]])
            mu, lv = net(f)
            loss = (0.5 * (lv + (h - mu) ** 2 / lv.exp())).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            mu, lv = net(Fva); h = Hva
            nll = (0.5 * (lv + (h - mu) ** 2 / lv.exp())).mean().item()
            samp = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            rms_real, rms_samp, rms_mu = h.pow(2).mean((0, 1, 3)).sqrt() * 10, samp.pow(2).mean((0, 1, 3)).sqrt() * 10, mu.pow(2).mean((0, 1, 3)).sqrt() * 10
            corr = torch.stack([torch.corrcoef(torch.stack([mu[..., k, :].reshape(-1), h[..., k, :].reshape(-1)]))[0, 1] for k in range(5)])
        print(f'ep {ep:3d} val NLL {nll:.3f}  HF rms real {np.round(rms_real.cpu().numpy(), 2)} sampled {np.round(rms_samp.cpu().numpy(), 2)} '
              f'predicted-mean {np.round(rms_mu.cpu().numpy(), 2)} corr(mu, real) {np.round(corr.cpu().numpy(), 2)}', flush=True)
        if nll < best:
            best = nll
            torch.save({'state_dict': net.state_dict(), 'fps': FPS, 'val_nll': nll}, args.out)
    json.dump(build_orientation_bank(body), open(args.out + '.orientation_bank.json', 'w'))
    print('saved', args.out, 'best val NLL', best)


# ----------------------------------------------------------------------------- C. orientation bank
def build_orientation_bank(body):
    """per real session and sensor: constant mounting offset C (rotvec) of the sensor orientation w.r.t. the reference bone and the rms jitter after removing it"""
    import mobileposer.articulate as art
    bones = [18, 19, 4, 5, 0]
    offs, jit = [], []
    for s in sequences(body, TRAIN):
        o_s, o_j = [], []
        for k, j in enumerate(bones):
            M = (s['ori'][:, k].transpose(-1, -2) @ s['G'][:, j]).mean(0)
            U, S, Vt = torch.linalg.svd(M)
            if torch.det(U @ Vt) < 0:
                U[:, -1] *= -1
            C = U @ Vt                                                           # ori_real @ C ~ bone  ->  ori_bone @ C^T ~ ori_real
            o_s.append(art.math.rotation_matrix_to_axis_angle(C.T.reshape(1, 3, 3)).reshape(3).tolist())
            res = s['ori'][:, k] @ C
            e = torch.rad2deg(torch.acos((((res.transpose(-1, -2) @ s['G'][:, j]).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
            o_j.append(float(e.pow(2).mean().sqrt()))
        offs.append(o_s); jit.append(o_j)
    return {'sensors': ['lw', 'rw', 'ls', 'rs', 'pelvis'], 'offset_rotvec': offs, 'jitter_deg_rms': jit}


# ----------------------------------------------------------------------------- apply to a cached synthetic dataset
def cmd_apply(args):
    import mobileposer.articulate as art
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.sim, map_location='cpu')
    net = HFSim().to(dev)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    bank = json.load(open(args.sim + '.orientation_bank.json'))
    offs = torch.tensor(bank['offset_rotvec']) * args.offset_scale                  # [N,5,3]
    jit = torch.tensor(bank['jitter_deg_rms']).mean(0) * np.pi / 180               # [5]
    data = torch.load(args.cache_in)
    imu = data['imu_inputs']
    n_win = len(imu)
    for i in range(n_win):
        x = imu[i].clone()
        G = data['pose_outputs'][i].reshape(-1, 24, 3, 3).float()
        J = data['joint_outputs'][i].reshape(-1, 24, 3).float()
        if len(G) < 8:
            continue
        with torch.no_grad():
            mu, lv = net(features(G, J).unsqueeze(0).to(dev))
            hf = (mu + (0.5 * lv).exp() * torch.randn_like(mu))[0].cpu() * 10 * args.hf_scale        # m/s^2
        x[:, :15] = x[:, :15] + (hf.reshape(len(x), 15) / 30.0)
        ori = x[:, 15:60].reshape(-1, 5, 3, 3)
        C = art.math.axis_angle_to_rotation_matrix(offs[torch.randint(len(offs), (1,))][0]).view(5, 3, 3)
        Jt = art.math.axis_angle_to_rotation_matrix(torch.randn(len(x) * 5, 3) * jit.repeat(len(x)).unsqueeze(-1)).view(-1, 5, 3, 3)
        x[:, 15:60] = (Jt @ ori @ C).reshape(len(x), 45)
        imu[i] = x
        if i % 5000 == 0:
            print('window', i, '/', n_win, flush=True)
    torch.save(data, args.cache_out + '.tmp')
    os.replace(args.cache_out + '.tmp', args.cache_out)
    print('wrote', args.cache_out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); t.add_argument('--epochs', type=int, default=60); t.add_argument('--out', default='checkpoints/hfsim.pt')
    a = sub.add_parser('apply')
    a.add_argument('--cache_in', required=True); a.add_argument('--cache_out', required=True); a.add_argument('--sim', default='checkpoints/hfsim.pt')
    a.add_argument('--hf_scale', type=float, default=1.0); a.add_argument('--offset_scale', type=float, default=1.0)
    args = ap.parse_args()
    {'train': cmd_train, 'apply': cmd_apply}[args.cmd](args)
