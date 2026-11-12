"""Diagnostic (no training): where is the baseline's error, by MOTION CATEGORY and
by JOINT TYPE (bone carrying an IMU vs not)?

Question behind it: for squat / lying / crawling-type motions the baseline is poor.
Is that because (a) the joints have no sensor (observability), (b) the training data
contains almost no such motions (AMASS: sitting 18, crawling 13, lying 5 of 9336 by
the keyword classification in data/classification_manifest.csv), or both?

Mapping: the layout datasets are built by no_head_layouts._load_amass_dataset, which
enumerates sorted(glob(raw_amass/<subset>/*/*_poses.npz)) and keeps files that load and
have framerate in {120,60,59}. We replay that enumeration to recover which raw file
each processed sequence is, then look the file up in the manifest. The count must match
len(data['pose']) per subset or the subset is skipped.

Error = ROOT-ALIGNED GLOBAL joint-angle error (SIP convention). Sequences from the
training subsets are SEEN by the model (optimistic); BMLhandball / BioMotionLab_NTroje
are held out. The 'seen' flag is reported.

Run from SparseIMU/:
    python -m mobileposer.diagnose_error_by_motion_category --manifest /tmp/classification_manifest.csv
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_drift_anchorpose import prepare_imu
from drift_eval_common import angle_between_rotmats

HOLDOUT = {'BMLhandball', 'BioMotionLab_NTroje'}
# layout wrists_shanks_waist: IMUs on both forearms, both shanks, waist(pelvis)
GROUPS = {
    'pelvis[0]': [0],
    'SENSED shank/knee[4,5]': [4, 5],
    'SENSED forearm/elbow[18,19]': [18, 19],
    'UNSENSED thigh/hip[1,2]': [1, 2],
    'UNSENSED spine[3,6,9]': [3, 6, 9],
    'UNSENSED upperarm/shoulder[16,17]': [16, 17],
    'UNSENSED neck/head/clav[12,13,14,15]': [12, 13, 14, 15],
}


def kept_files(subset):
    out = []
    for f in sorted(glob.glob(os.path.join(paths.raw_amass, subset, '*/*_poses.npz'))):
        try:
            c = np.load(f)
            fr = int(c['mocap_framerate'])
        except Exception:
            continue
        if fr not in [120, 60, 59]:
            continue
        out.append(f)
    return out


def load_manifest(path):
    m = {}
    with open(path, encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            raw = row['raw_motion'].replace('\\', '/')
            key = '/'.join(raw.split('/AMASS/')[-1].split('/')) if '/AMASS/' in raw else raw
            m[key] = row['category']
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data-root', default='data/no_head_5imu_surface_processed/wrists_shanks_waist')
    ap.add_argument('--checkpoint', default=str(paths.root_dir / 'checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth'))
    ap.add_argument('--max-per-category', type=int, default=40)
    ap.add_argument('--max-frames', type=int, default=3000)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device).eval()
    body = ParametricModel(paths.smpl_file)
    manifest = load_manifest(args.manifest)

    picked = defaultdict(list)       # category -> [(subset, idx, file)]
    totals = defaultdict(int)
    for pt in sorted(Path(args.data_root).glob('*.pt')):
        subset = pt.stem
        files = kept_files(subset)
        d = torch.load(pt, map_location='cpu', weights_only=False)
        n = len(d['pose'])
        if len(files) != n:
            print(f'[skip] {subset}: replayed {len(files)} files vs {n} sequences'); continue
        for i, f in enumerate(files):
            rel = os.path.relpath(f, paths.raw_amass).replace('\\', '/')
            cat = manifest.get(rel, 'unlisted')
            totals[cat] += 1
            if len(picked[cat]) < args.max_per_category and d['pose'][i].shape[0] >= 90:
                picked[cat].append((subset, i, d['acc'][i], d['ori'][i], d['pose'][i]))
        del d
    print('sequence counts per category in processed data:', dict(totals))

    rows = []
    for cat, items in sorted(picked.items(), key=lambda kv: -len(kv[1])):
        per_joint = []
        frames = 0
        seen = 0
        for subset, i, acc, ori, pose in items:
            T = min(len(pose), args.max_frames)
            gt = pose[:T].reshape(T, 24, 3, 3)
            with torch.no_grad():
                imu = prepare_imu(acc[:T], ori[:T]).to(device)
                model.reset()
                pred, _, _, _ = model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
                pred = (pred.squeeze(0) if pred.dim() == 4 else pred).cpu().reshape(-1, 24, 3, 3)
                T2 = min(T, len(pred))
                pp, gg = pred[:T2].clone(), gt[:T2].clone()
                pp[:, 0] = torch.eye(3)
                gg[:, 0] = torch.eye(3)
                gp, _ = body.forward_kinematics(pp, torch.ones(10), tran=None, calc_mesh=False)
                gq, _ = body.forward_kinematics(gg, torch.ones(10), tran=None, calc_mesh=False)
                e = angle_between_rotmats(gp.reshape(T2, 24, 3, 3), gq.reshape(T2, 24, 3, 3))   # [T,24]
                e0 = angle_between_rotmats(pred[:T2, 0], gt[:T2, 0])                              # pelvis, unaligned
                e[:, 0] = e0
            per_joint.append(e.mean(0).numpy())
            frames += T2
            seen += int(subset not in HOLDOUT)
        pj = np.mean(per_joint, axis=0)
        rows.append((cat, len(items), totals[cat], frames, seen, {g: float(pj[js].mean()) for g, js in GROUPS.items()}))

    names = list(GROUPS)
    print('\ncategory       n_eval/total  frames  seen   ' + '  '.join(n[:22].rjust(22) for n in names))
    for cat, n, tot, fr, seen, g in rows:
        print(f'{cat:14s} {n:4d}/{tot:<6d} {fr:7d} {seen:4d}   ' + '  '.join(f'{g[nm]:22.2f}' for nm in names))
    print('\n(deg; root-aligned global; pelvis column is the unaligned root error. seen = #eval sequences from training subsets)')


if __name__ == '__main__':
    main()
