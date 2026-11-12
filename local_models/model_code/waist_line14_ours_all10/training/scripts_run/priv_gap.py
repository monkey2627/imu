"""Does the privileged teacher (5 IMUs + 7 virtual IMUs) estimate joint positions better than the 5-IMU student?
Compares, on AMASS windows, the position error (cm) of: baseline Joints, jpriv student, jpriv teacher.
NOTE: all AMASS windows were used for training (the 10% validation split is not reproducible), so this is an in-sample comparison.
run (SparseIMU/): MOBILEPOSER_FPS=60 MOBILEPOSER_PRIV=1 python scripts_run/priv_gap.py --priv pnp5_60a_s42_jpriv_joints --base pnp5_60a_s42_joints
"""
import argparse, os, sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from mobileposer.data import PoseDataset, pad_seq
from mobileposer.finetune_dip import best_ckpt
from mobileposer.models.joints import Joints

ap = argparse.ArgumentParser()
ap.add_argument('--priv', required=True); ap.add_argument('--base', required=True)
ap.add_argument('--cache', default='data/cache_pnp5_60a.pt'); ap.add_argument('--n', type=int, default=3000)
a = ap.parse_args()
dev = torch.device('cuda:0')

ds = PoseDataset.__new__(PoseDataset)
ds.evaluate, ds.finetune, ds.data = None, None, torch.load(a.cache)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:a.n].tolist()
sub = torch.utils.data.Subset(ds, idx)

priv = Joints.load_from_checkpoint(best_ckpt(f'checkpoints/{a.priv}'), map_location=dev).to(dev).eval()
os.environ['MOBILEPOSER_PRIV'] = '0'
base = Joints.load_from_checkpoint(best_ckpt(f'checkpoints/{a.base}'), map_location=dev).to(dev).eval()

groups = {'all': list(range(1, 24)), 'arms': [13, 14, 16, 17, 18, 19, 20, 21, 22, 23], 'legs': [1, 2, 4, 5, 7, 8, 10, 11], 'spine+head': [3, 6, 9, 12, 15]}
acc = {k: {g_: [0.0, 0] for g_ in groups} for k in ('baseline student', 'jpriv student', 'jpriv teacher')}


def add(name, pred, tgt, mask):
    err = (pred.view(*pred.shape[:2], 24, 3) - tgt.view(*tgt.shape[:2], 24, 3)).norm(dim=-1) * 100      # cm [B,S,24]
    for g_, js in groups.items():
        e = err[:, :, js].mean(-1)[mask]
        acc[name][g_][0] += e.sum().item(); acc[name][g_][1] += e.numel()


with torch.no_grad():
    for (imu, lens), (out, _) in DataLoader(sub, batch_size=32, collate_fn=pad_seq):
        imu = imu.to(dev); out = {k: v.to(dev) for k, v in out.items()}
        S = imu.shape[1]
        mask = torch.arange(S, device=dev)[None] < torch.as_tensor(lens, device=dev)[:, None]
        tgt = out['joints']
        add('baseline student', base(imu, lens), tgt, mask)
        add('jpriv student', priv(imu, lens), tgt, mask)
        t_raw, _ = priv._run(priv.teacher, torch.cat([imu, priv._priv_inputs(out)], -1), lens)
        add('jpriv teacher', priv._post(t_raw), tgt, mask)

print(f'{"mean joint position error (cm), root-relative":<48}' + ''.join(f'{g_:>12}' for g_ in groups))
for k, v in acc.items():
    print(f'{k:<48}' + ''.join(f'{v[g_][0] / v[g_][1]:>12.2f}' for g_ in groups))
