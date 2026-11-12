"""INNOVATION 2 - self-calibrating sparse-IMU pose estimation (no T-pose, no sensor-to-bone calibration step).

Pose networks assume that the measured sensor orientation IS the orientation of the bone.  In real use there is a constant,
unknown mounting offset C_k per sensor (sensor frame vs bone frame): TotalCapture's official calibration is ~12 deg off, DIP's ~8.6 deg,
and on our own hardware the as-given sensed bones were 20-30 deg off.  Pose networks trained with offset noise only learn to
trust the inputs less (measured: lower precision on clean data).  CalibNet instead ESTIMATES the offsets from the motion itself
and removes them, so the pose network can keep trusting the (corrected) orientations.

Observability: the offset is a constant over a session, and the motion constrains it - gravity direction in the bone frame during
standing / walking, the swing plane of the legs and arms (functional calibration), the plausible relative orientation between
bones.  CalibNet is a bidirectional GRU over a window of orientations and accelerations expressed in the pelvis HEADING frame
(heading invariant, gravity kept), mean-pooled, regressing the 6D rotation of every sensor's offset.  Training is supervised
and free of real labels: synthetic AMASS windows are corrupted with KNOWN random offsets (ori_meas = ori_bone @ C).

  python -m mobileposer.calibnet train --cache data/cache_pnp5_60.pt --steps 30000 --out checkpoints/calibnet.pt
  python -m mobileposer.calibnet eval  --ckpt checkpoints/calibnet.pt          # offsets of the real DIP / TotalCapture sets: estimated vs oracle
  (evaluate_standard.py --calib checkpoints/calibnet.pt [--calib_mode whole|first10] corrects the inputs before the pose network)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

import mobileposer.articulate as art
from mobileposer.frames import frame_rot

FPS = int(os.environ.get('MOBILEPOSER_FPS', 30))
BONE_JOINT = [18, 19, 4, 5, 0]               # bone each sensor of the 5-IMU layout sits on (pelvis sensor: joint 0)
NS = 5


def to_rot(r6):
    """6D -> rotation matrix (Gram-Schmidt), [...,6] -> [...,3,3]"""
    a, b = r6[..., :3], r6[..., 3:]
    x = torch.nn.functional.normalize(a, dim=-1)
    z = torch.nn.functional.normalize(torch.cross(x, b, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def geodesic(R1, R2):
    c = ((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


class CalibNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(NS * 12, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, NS * 6))
        self.register_buffer('id6', torch.tensor([1.0, 0, 0, 0, 1.0, 0]).repeat(NS))

    def forward(self, acc, ori):
        """acc [B,T,5,3] (world, m/s^2 / 30), ori [B,T,5,3,3] (world, measured) -> offsets C_hat [B,5,3,3] with ori_meas = ori_bone @ C"""
        B, T = acc.shape[:2]
        Rf = frame_rot(ori[:, :, NS - 1].reshape(-1, 3, 3), 'yaw').reshape(B, T, 3, 3)          # pelvis heading frame
        RfT = Rf.transpose(-1, -2).unsqueeze(2)
        a = (RfT @ acc.unsqueeze(-1)).squeeze(-1)
        o = RfT @ ori
        h, _ = self.gru(self.inp(torch.cat([a.flatten(2), o.flatten(2)], dim=-1)))
        r6 = self.head(h.mean(1)) + self.id6
        return to_rot(r6.reshape(B, NS, 6))


def random_offsets(B, max_deg=30.0):
    """B x 5 random rotations: uniform axis, angle |N(0, s)| with s skewed towards small values (0.5 .. max_deg deg)"""
    s = (0.5 + torch.rand(B, 1) ** 2 * (max_deg - 0.5)) * np.pi / 180        # skewed to small offsets: calibrated data (DIP, ~1 deg) must stay untouched
    axis = torch.nn.functional.normalize(torch.randn(B, NS, 3), dim=-1)
    ang = torch.randn(B, NS).abs() * s
    return art.math.axis_angle_to_rotation_matrix((axis * ang.unsqueeze(-1)).reshape(-1, 3)).reshape(B, NS, 3, 3)


def cmd_train(args):
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.load(args.cache)['imu_inputs']
    L = args.window
    pool = [x for x in data if len(x) >= L]
    print('windows >=', L, ':', len(pool), flush=True)
    net = CalibNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    run = 0.0
    for step in range(args.steps):
        idx = torch.randint(len(pool), (args.batch,))
        x = torch.stack([pool[i][:L] for i in idx.tolist()])
        acc = x[..., :15].reshape(-1, L, NS, 3).to(dev)
        ori = x[..., 15:60].reshape(-1, L, NS, 3, 3).to(dev)
        C = random_offsets(args.batch, args.max_deg).to(dev)
        jit = art.math.axis_angle_to_rotation_matrix(torch.randn(args.batch * L * NS, 3, device=dev) * 0.01745).reshape(args.batch, L, NS, 3, 3)
        ori_m = jit @ ori @ C.unsqueeze(1)
        acc_m = acc + 0.01 * torch.randn_like(acc)                                              # ~0.3 m/s^2 accelerometer noise
        pred = net(acc_m, ori_m)
        loss = geodesic(pred, C).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 500 == 0:
            print(f'step {step:6d}  mean offset-estimation error {np.degrees(run):.2f} deg (identity baseline: {np.degrees(geodesic(torch.eye(3, device=dev).expand_as(C), C).mean().item()):.2f})', flush=True)
        if step % 2000 == 0 or step == args.steps - 1:
            torch.save({'state_dict': net.state_dict(), 'window': L, 'fps': FPS}, args.out)
    print('saved', args.out)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    net = CalibNet().to(device)
    net.load_state_dict(ck['state_dict'])
    return net.eval(), ck['window']


def chordal_mean(Rs):
    """[N,5,3,3] -> [5,3,3] rotation average"""
    M = Rs.mean(0)
    U, S, Vt = torch.linalg.svd(M)
    d = torch.det(U @ Vt)
    U = U.clone()
    U[..., :, -1] *= d.unsqueeze(-1)
    return U @ Vt


@torch.no_grad()
def estimate(net, window, acc, ori, upto=None, stride=None, device='cpu'):
    """session offsets from the first `upto` frames (None = all): average of window estimates. acc [T,5,3] (/30 scaled), ori [T,5,3,3]"""
    T = len(acc) if upto is None else min(upto, len(acc))
    L = min(window, T)
    stride = stride or max(1, L // 2)
    starts = list(range(0, max(1, T - L + 1), stride))
    A = torch.stack([acc[s:s + L] for s in starts]).to(device)
    O = torch.stack([ori[s:s + L] for s in starts]).to(device)
    return chordal_mean(net(A, O)).cpu()


def correct(ori, C):
    """ori_meas = ori_bone @ C  ->  ori_bone = ori_meas @ C^T"""
    return ori @ C.transpose(-1, -2)


def cmd_eval(args):
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    from mobileposer.evaluate_standard import sequences, load_pnp_articulate, DATASETS
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net, window = load(args.ckpt, dev)
    body = ParametricModel(paths.smpl_file)
    art2 = load_pnp_articulate()
    print('mean angle (deg) of the oracle offset (what an uncorrected network sees) and of the residual after CalibNet, per sensor [lw rw ls rs pelvis]')
    for ds in DATASETS:
        base, resid = [], []
        for name, a, R, pose in sequences(ds, art2):
            G, _ = body.forward_kinematics(pose, torch.zeros(10), tran=None, calc_mesh=False)
            G = G.reshape(-1, 24, 3, 3)
            Co = []
            for k, j in enumerate(BONE_JOINT):                                  # oracle: R_meas = G_bone @ C  ->  C = mean(G^T R)
                M = (G[:, j].transpose(-1, -2) @ R[:, k]).mean(0)
                U, S, Vt = torch.linalg.svd(M)
                if torch.det(U @ Vt) < 0:
                    U[:, -1] *= -1
                Co.append(U @ Vt)
            Co = torch.stack(Co)
            Ce = estimate(net, window, a / 30.0, R, upto=args.upto, device=dev)
            base.append(np.degrees(geodesic(torch.eye(3).expand_as(Co), Co).numpy()))
            resid.append(np.degrees(geodesic(Ce, Co).numpy()))
        print(f'{ds:26s} oracle {np.round(np.mean(base, 0), 1)}   after CalibNet {np.round(np.mean(resid, 0), 1)}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--cache', required=True); t.add_argument('--out', default='checkpoints/calibnet.pt')
    t.add_argument('--steps', type=int, default=30000); t.add_argument('--batch', type=int, default=64)
    t.add_argument('--window', type=int, default=250); t.add_argument('--max_deg', type=float, default=30.0)
    e = sub.add_parser('eval')
    e.add_argument('--ckpt', default='checkpoints/calibnet.pt'); e.add_argument('--upto', type=int, default=None)
    a = ap.parse_args()
    {'train': cmd_train, 'eval': cmd_eval}[a.cmd](a)
