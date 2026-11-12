"""Whole-sequence storage for streaming training: raw acc/ori + root-relative targets, concatenated, kept on the GPU.
Random crops start at an arbitrary frame with a zero recurrent state, so the model learns to cold-start mid-motion.

  python -m mobileposer.stream.data --out data/stream_pnp5_60a.pt            # AMASS (pnp5_60a, 18 subsets)
  python -m mobileposer.stream.data --dip --out data/stream_dip_train.pt     # DIP-IMU s_01..s_08 (real IMU)
"""
import argparse
import os
from pathlib import Path

import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.stream.features import targets, imu_features, random_mount

VAL_EVERY = 50          # every 50th AMASS sequence is held out for validation


def _pack(seqs, body, dev):
    acc, ori, jrel, r6d, lens, names = [], [], [], [], [], []
    for name, a, R, pose in seqs:
        j, r = targets(body, pose.to(dev))
        acc.append(a.reshape(len(a), 15)); ori.append(R.reshape(len(R), 45))
        jrel.append(j.cpu()); r6d.append(r.cpu()); lens.append(len(a)); names.append(name)
    return {'acc': torch.cat(acc), 'ori': torch.cat(ori), 'jrel': torch.cat(jrel), 'r6d': torch.cat(r6d),
            'lens': torch.tensor(lens), 'names': names}


def amass_sequences(folder, holdout):
    k = 0
    for f in sorted(Path(folder).glob('*.pt')):
        d = torch.load(f)
        for i in range(len(d['pose'])):
            if len(d['pose'][i]) >= 120 and ((k % VAL_EVERY == 0) == holdout):
                yield f'{f.stem}/{i}', d['acc'][i][:, :5].float(), d['ori'][i][:, :5].float(), d['pose'][i].float()
            k += 1


def dip_sequences(subjects):
    from mobileposer.evaluate_dip_real import load_dip
    for s in load_dip(ParametricModel(paths.smpl_file), subjects=subjects):
        yield s['name'], s['acc'], s['ori'], s['pose']


class Store:
    """Concatenated sequences on `dev`; sample(B, L) returns padded crops (acc, ori, jrel, r6d, mask)."""
    def __init__(self, path, dev):
        d = torch.load(path)
        self.dev = dev
        for k in ('acc', 'ori', 'jrel', 'r6d'):
            setattr(self, k, d[k].to(dev))
        self.lens = d['lens'].to(dev)
        self.offs = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), self.lens.cumsum(0)[:-1]])
        self.names = d['names']

    def __len__(self):
        return len(self.lens)

    def sample(self, B, L):
        seq = torch.multinomial(self.lens.float(), B, replacement=True)
        n = self.lens[seq]
        start = (torch.rand(B, device=self.dev) * (n - L).clamp(min=0).float()).long()
        t = torch.arange(L, device=self.dev)
        mask = t[None] < n[:, None]
        idx = self.offs[seq, None] + (start[:, None] + t[None]).clamp(max=(n - 1)[:, None])
        return (self.acc[idx].view(B, L, 5, 3), self.ori[idx].view(B, L, 5, 3, 3),
                self.jrel[idx], self.r6d[idx], mask)


def batch_inputs(acc, ori, gyro, mount_deg=0.0, frame='rel', local=False):
    """-> features [B,T,F] and the (possibly mount-perturbed) orientations the model actually sees"""
    if mount_deg > 0:
        ori = ori @ random_mount(len(ori), mount_deg, ori.device).unsqueeze(1)
    return imu_features(acc, ori, gyro, frame, local), ori


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--amass', default='data/pnp5_60a/pnp5')
    ap.add_argument('--dip', action='store_true')
    a = ap.parse_args()
    assert int(os.environ.get('MOBILEPOSER_FPS', 30)) == 60, 'set MOBILEPOSER_FPS=60 (load_dip frame step)'
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file, device=dev)
    if a.dip:
        out = _pack(dip_sequences([f's_{i:02d}' for i in range(1, 9)]), body, dev)
        torch.save(out, a.out)
        print('dip', len(out['lens']), 'sequences', int(out['lens'].sum()), 'frames')
    else:
        for holdout, path in ((False, a.out), (True, a.out.replace('.pt', '_val.pt'))):
            out = _pack(amass_sequences(a.amass, holdout), body, dev)
            torch.save(out, path)
            print(path, len(out['lens']), 'sequences', int(out['lens'].sum()), 'frames', flush=True)
