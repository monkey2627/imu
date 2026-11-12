"""DIP-train fine-tuning for the line14 causal belief filter."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from mobileposer.articulate import math as M
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths, joint_set
from mobileposer.line14.belief import CausalBeliefFilter, belief_loss


class Windows(Dataset):
    def __init__(self, rows, parent, window=250):
        self.items = []
        for acc, ori, pose in rows:
            T = len(pose)
            local = pose
            glob = M.forward_kinematics_R(local, parent)
            target = M.rotation_matrix_to_r6d(glob[:, joint_set.reduced]).reshape(T, 96)
            imu = torch.cat((acc[:, [0, 1, 2, 3, 5]].reshape(T, 15) / 30.0,
                             ori[:, [0, 1, 2, 3, 5]].reshape(T, 45)), dim=-1)
            for s in range(0, T, window):
                self.items.append((imu[s:s + window].float(), target[s:s + window].float()))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


class Replay(Dataset):
    def __init__(self, dip, cache, ratio=3):
        self.dip, self.cache, self.ratio = dip, cache, int(ratio)
    def __len__(self): return len(self.dip) * (1 + self.ratio)
    def __getitem__(self, i):
        if i < len(self.dip):
            return self.dip[i]
        j = int(torch.randint(len(self.cache['imu_inputs']), (1,)))
        imu = self.cache['imu_inputs'][j].float()
        glob = self.cache['pose_outputs'][j].float()
        target = M.rotation_matrix_to_r6d(glob[:, joint_set.reduced].reshape(-1, 3, 3)).reshape(len(imu), 96)
        return imu, target


def collate(rows):
    n = max(len(x[0]) for x in rows)
    imu = rows[0][0].new_zeros(len(rows), n, 60)
    tgt = rows[0][1].new_zeros(len(rows), n, 96)
    lengths = []
    for i, (x, y) in enumerate(rows):
        imu[i, :len(x)] = x; tgt[i, :len(y)] = y; lengths.append(len(x))
    return imu, tgt, torch.tensor(lengths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='experiments/line14/belief_filter/checkpoints/belief_s42_full.pt')
    ap.add_argument('--data', default='/home/duanyuhan/dyh/motion/MotionRe/base_mobileposer/data/processed_datasets/eval/dip_train.pt')
    ap.add_argument('--out', default='experiments/line14/belief_filter/checkpoints/belief_s42_dipft.pt')
    ap.add_argument('--replay-cache', default='data/cache_pnp5_60a.pt')
    ap.add_argument('--sip-weight', type=float, default=1.0,
                    help='extra pose-loss weight for reduced joints [1,2,7,8], corresponding to SIP hips/shoulders')
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args(); dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    d = torch.load(a.data, map_location='cpu')
    rows = list(zip(d['acc'], d['ori'], d['pose']))
    body = ParametricModel(paths.smpl_file, device=torch.device('cpu'))
    dip = Windows(rows, body.parent)
    cache = torch.load(a.replay_cache, map_location='cpu')
    ds = Replay(dip, cache, ratio=3); dl = DataLoader(ds, batch_size=a.batch, shuffle=True, collate_fn=collate)
    ck = torch.load(a.init, map_location='cpu'); model = CausalBeliefFilter(hidden=int(ck.get('hidden', 256))).to(dev)
    model.load_state_dict(ck['model']); opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    history = []
    for ep in range(a.epochs):
        model.train(); total = 0.0
        for imu, tgt, lengths in dl:
            imu, tgt = imu.to(dev), tgt.to(dev); valid = torch.arange(imu.shape[1], device=dev)[None] < lengths.to(dev)[:, None]
            opt.zero_grad(set_to_none=True); out = model(imu)
            loss = belief_loss(out, imu, tgt, pose_weight=0.0, mask=valid)
            pose = out['pose'].reshape(*out['pose'].shape[:2], 16, 6)
            target = tgt.reshape(*tgt.shape[:2], 16, 6)
            joint_weight = torch.ones(16, device=dev)
            joint_weight[[1, 2, 7, 8]] = 1.0 + a.sip_weight
            pose_error = (pose - target).square().mean(-1)
            weights = valid.to(pose_error.dtype).unsqueeze(-1) * joint_weight.view(1, 1, 16)
            loss = loss + (pose_error * weights).sum() / weights.sum().clamp_min(1.0)
            if not torch.isfinite(loss): raise FloatingPointError('non-finite DIP fine-tune loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += float(loss.detach())
        row = {'epoch': ep + 1, 'train_loss': total / max(len(dl), 1)}; history.append(row); print(row, flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'hidden': model.hidden, 'history': history,
                'future_frames': 0, 'calibration': 'none', 'sip_weight': a.sip_weight,
                'fine_tune': 'DIP train subjects s01-s08 + AMASS replay x3'}, out)
    print({'saved': str(out), 'epochs': a.epochs, 'dip_windows': len(dip), 'mixed_windows': len(ds),
           'sip_weight': a.sip_weight}, flush=True)


if __name__ == '__main__': main()
