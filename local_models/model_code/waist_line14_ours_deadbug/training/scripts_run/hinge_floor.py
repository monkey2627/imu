"""How much error does the hinge constraint itself introduce on GT poses?
For the true joint rotation E = R_parent^T R_child (elbow: joints 16->18 / 17->19, knee: 1->4 / 2->5) fit
  elbow: Rot(a0, t0) Rot(a1, t1)      knee: Rot(a2, t)
with (a) the initial axes used in models/poser.py and (b) axes fitted on the data, per-frame angles free.
Reports the residual geodesic angle (deg) = the floor the constraint puts under the thigh / upper-arm orientation.
run (SparseIMU/): python scripts_run/hinge_floor.py
"""
import math, os, sys
import torch

sys.path.insert(0, os.getcwd())
from mobileposer.data import PoseDataset

dev = torch.device('cuda:0')
ds = PoseDataset.__new__(PoseDataset)
ds.evaluate, ds.finetune, ds.data = None, None, torch.load('data/cache_pnp5_60a.pt')
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:1500].tolist()
P = torch.cat([ds.data['pose_outputs'][i][::5] for i in idx]).to(dev)                  # global rotation matrices [N,24,3,3]
print('frames', len(P), flush=True)


def rot(a, t):
    a = a / a.norm()
    z = torch.zeros((), device=a.device)
    K = torch.stack([z, -a[2], a[1], a[2], z, -a[0], -a[1], a[0], z]).view(3, 3)
    return torch.eye(3, device=a.device) + t.sin()[:, None, None] * K + (1 - t.cos())[:, None, None] * (K @ K)


def ang(A, B):
    c = ((A.transpose(-1, -2) @ B).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / math.pi


def fit(E, two, init_axes, fit_axes, steps=400):
    axes = torch.nn.Parameter(torch.tensor(init_axes, device=dev, dtype=torch.float32), requires_grad=fit_axes)
    t = torch.nn.Parameter(torch.zeros(len(E), 2, device=dev))
    opt = torch.optim.Adam([{'params': [t], 'lr': 0.1}] + ([{'params': [axes], 'lr': 0.02}] if fit_axes else []))
    for _ in range(steps):
        R = rot(axes[0], t[:, 0]) @ (rot(axes[1], t[:, 1]) if two else torch.eye(3, device=dev))
        loss = ang(R, E).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    R = rot(axes[0], t[:, 0]) @ (rot(axes[1], t[:, 1]) if two else torch.eye(3, device=dev))
    return ang(R, E).mean().item(), axes.detach().cpu().numpy().round(2).tolist()


for name, (par, ch, two, init) in {'L elbow': (16, 18, True, [[0, 1, 0], [1, 0, 0]]), 'R elbow': (17, 19, True, [[0, 1, 0], [1, 0, 0]]),
                                   'L knee': (1, 4, False, [[1, 0, 0]]), 'R knee': (2, 5, False, [[1, 0, 0]])}.items():
    E = P[:, par].transpose(-1, -2) @ P[:, ch]
    spread = ang(E, torch.eye(3, device=dev).expand_as(E)).mean().item()
    r0, _ = fit(E, two, init, False)
    r1, ax = fit(E, two, init, True)
    print(f'{name}: mean |E| {spread:5.1f} deg | residual with initial axes {r0:5.2f} deg | with fitted axes {r1:5.2f} deg  axes {ax}', flush=True)
