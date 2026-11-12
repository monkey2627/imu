"""E4: frequency-band / per-joint decomposition of the joint-position error (validation only).

For each model the per-frame position error vector (cm) of every validation sequence is stitched from the
consecutive non-overlapping 125-frame windows (PairedWindowDataset enumerates them in order), then split by FFT
into  DC (sequence-mean bias) | 0<f<0.2 Hz | mid (0.2-1 Hz) | high (>=1 Hz)  at 60 fps.
Energy is exact (Parseval).  Only sequences >= MIN_FRAMES (20 s) are used so that 0.2 Hz is resolvable.
Reports RMS error (cm) per joint group and band, 3 seeds pooled (mean of squared errors).
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from mobileposer.line9.models import Line9Joints, GeometryStats
from mobileposer.line9.evaluation import JOINT_GROUPS
from mobileposer.line14.geom import GeomProvider
from mobileposer.line14.train_common import build_val, val_loader

FPS = 60
MIN_FRAMES = 1200
BANDS = (("DC(seq-mean)", -1.0, 1e-9), ("0<f<0.2Hz", 1e-9, 0.2), ("mid0.2-1Hz", 0.2, 1.0), ("high>=1Hz", 1.0, 1e9))
S9 = Path('/home/duanyuhan/dyh/motion/SparseIMU/checkpoints/line9_shape')
VARIANTS = [  # label, ckpt pattern, geometry kind (None = B0)
    ("B0", str(S9 / 'b0_s{s}.pt'), None),
    ("R1", 'checkpoints/line14/oracle_s{s}.pt', 'r1'),
    ("height", 'checkpoints/line14/E2/height_oracle_s{s}.pt', 'height'),
    ("arms", 'checkpoints/line14/E2/arms_oracle_s{s}.pt', 'arms'),
    ("legs", 'checkpoints/line14/E2/legs_oracle_s{s}.pt', 'legs'),
    ("R1_n5", 'checkpoints/line14/E3train/r1_oracle_n5i_s{s}.pt', 'r1'),
    ("R1_n10", 'checkpoints/line14/E3train/r1_oracle_n10i_s{s}.pt', 'r1'),
]


def load(label, pattern, kind, seed, device):
    st = torch.load(pattern.format(s=seed), map_location=device, weights_only=False)
    if kind is None:
        m = Line9Joints(mode='b0', geom_dim=33)
    else:
        m = Line9Joints(mode='oracle', geom_dim=GeomProvider(kind).dim, geometry_stats=GeometryStats(mean=st['geometry_mean'], std=st['geometry_std']))
    m.load_state_dict(st['model']); return m.to(device).eval()


def band_sq_energy(err):  # err [N,24,3] cm -> {band: [24] sum of squared error (sum over frames and xyz)}
    N = err.shape[0]
    X = np.fft.rfft(err, axis=0)
    freq = np.fft.rfftfreq(N, d=1.0 / FPS)
    w = np.full(len(freq), 2.0); w[0] = 1.0
    if N % 2 == 0: w[-1] = 1.0
    P = (np.abs(X) ** 2).sum(-1) * w[:, None] / N        # [F,24]
    return {name: P[(freq >= lo) & (freq < hi)].sum(0) for name, lo, hi in BANDS}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--seeds', default='42,7,3')
    ap.add_argument('--out', type=Path, default=Path('results/line14/E4_bands.json')); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    min_frames = 100 if a.smoke else MIN_FRAMES
    records, val = build_val(smoke=a.smoke)
    provs = {k: GeomProvider(k) for k in ('r1', 'height', 'arms', 'legs')}
    energy = {v[0]: {b[0]: np.zeros(24) for b in BANDS} for v in VARIANTS}; frames = 0; nseq = 0
    for seed in [int(x) for x in a.seeds.split(',')]:
        models = {label: load(label, pat, kind, seed, device) for label, pat, kind in VARIANTS}
        store = {label: defaultdict(list) for label in models}; keys = {}
        with torch.no_grad():
            for b in val_loader(val):
                x = b['target_imu'].to(device); gt = b['target_joints']
                for label, pat, kind in VARIANTS:
                    g = None if kind is None else provs[kind].batch(b['beta'], b['identity']).to(device)
                    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                        p = models[label](x, b['target_lengths'].tolist(), g)
                    err = (p.float().cpu().reshape(*p.shape[:2], 24, 3) - gt.reshape(*gt.shape[:2], 24, 3)) * 100.0
                    for i in range(err.shape[0]):
                        n = int(b['target_lengths'][i]); k = (b['identity'][i], b['target_subset'][i], b['target_sequence'][i])
                        store[label][k].append((int(b['window_start'][i]), err[i, :n].numpy()))
        used = 0; used_frames = 0
        for k in store['B0']:
            parts = sorted(store['B0'][k], key=lambda t: t[0])
            if sum(len(e) for _, e in parts) < min_frames: continue
            used += 1; used_frames += sum(len(e) for _, e in parts)
            for label in store:
                seq = np.concatenate([e for _, e in sorted(store[label][k], key=lambda t: t[0])], 0).astype(np.float64)
                for bn, en in band_sq_energy(seq).items(): energy[label][bn] += en
        frames += used_frames; nseq += used
        print(f'seed {seed}: used {used} sequences, {used_frames} frames', flush=True)
    out = {'min_frames': min_frames, 'sequences_used_all_seeds': nseq, 'frames_all_seeds': frames, 'rms_cm': {}}
    for label in energy:
        out['rms_cm'][label] = {}
        for bn in energy[label]:
            e = energy[label][bn]
            out['rms_cm'][label][bn] = {gname: float(np.sqrt(e[list(idx)].sum() / (frames * len(idx)))) for gname, idx in JOINT_GROUPS.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    for gname in JOINT_GROUPS:
        print(f'== {gname}: RMS cm [DC, 0-0.2, 0.2-1, >=1]')
        for label in energy:
            print('  %-8s' % label, ' '.join('%6.3f' % out['rms_cm'][label][bn][gname] for bn, _, _ in BANDS),
                  ' total %.3f' % np.sqrt(sum(out['rms_cm'][label][bn][gname] ** 2 for bn, _, _ in BANDS)))


if __name__ == '__main__':
    main()
