"""Primary-protocol comparison of method variants (all trained on v3, world-frame inputs, bidirectional):
  base   : v3_base            (original MobilePoser training)
  dir    : bone-direction loss in Poser                       (Joints = v3_base)
  relacc : explicit relative-acceleration features in Poser+Joints
  both   : relacc features + direction loss                   (Joints = relacc)
Test 1 (primary)      : DIP-IMU test split, real IMU, unaligned global orientation error per group + bone DIRECTION error.
Test 2 (same-dist.)   : 68 held-out AMASS stand->hold events, thigh error (deg) by time since flexion onset, full-sequence inference.

Run from SparseIMU/:  python -m mobileposer.evaluate_method_variants
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_v2_models import ModulePredictor
from mobileposer.evaluate_canon_models import GROUPS
from mobileposer.evaluate_dip_real import load_dip
from mobileposer.evaluate_causal_holds import BINS
from mobileposer.diagnose_dip_direction_vs_twist import vec_angle
from drift_eval_common import angle_between_rotmats

VARIANTS = {'base': ('v3_base_poser', 'v3_base_joints', False),
            'dir': ('v3_dir_poser', 'v3_base_joints', False),
            'relacc': ('v3_relacc_poser', 'v3_relacc_joints', True),
            'both': ('v3_both_poser', 'v3_relacc_joints', True),
            'geo': ('v3_geo_poser', 'v3_base_joints', False),
            'off': ('v3_off_poser', 'v3_off_joints', False),
            'base_s2': ('v3_base_s2_poser', 'v3_base_s2_joints', False),
            'relacc_s2': ('v3_relacc_s2_poser', 'v3_relacc_s2_joints', True)}


def load(poser, joints, relacc, dev):
    if relacc:
        os.environ['MOBILEPOSER_RELACC'] = '1'
    try:
        return ModulePredictor(f'checkpoints/{poser}', f'checkpoints/{joints}', False, dev)
    finally:
        os.environ.pop('MOBILEPOSER_RELACC', None)


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    models = {}
    for k, (p, j, r) in VARIANTS.items():
        if Path(f'checkpoints/{p}').exists() and Path(f'checkpoints/{j}').exists():
            models[k] = load(p, j, r, dev)
    import mobileposer.articulate as art

    def dip_eval(title, offset_deg=0.0):
        res, wts = {k: [] for k in models}, []
        for si, s in enumerate(load_dip(body)):
            T = len(s['pose'])
            ori = s['ori']
            if offset_deg > 0:                              # same random constant sensor-to-bone offsets for every model
                g = torch.Generator().manual_seed(1000 + si)
                axis = torch.nn.functional.normalize(torch.randn(5, 3, generator=g), dim=-1)
                ang = torch.randn(5, generator=g).abs() * (offset_deg * np.pi / 180)
                C = art.math.axis_angle_to_rotation_matrix(axis * ang.unsqueeze(-1)).view(5, 3, 3)
                ori = ori @ C
            G, J = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
            G, J = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
            wts.append(T)
            for k, fn in models.items():
                Gp, Jp = body.forward_kinematics(fn(s['acc'], ori)[:T], torch.ones(10), tran=None, calc_mesh=False)
                Gp, Jp = Gp.reshape(T, 24, 3, 3), Jp.reshape(T, 24, 3)
                e = angle_between_rotmats(Gp, G).mean(0).numpy()
                d_arm = (vec_angle(J[:, 18] - J[:, 16], Jp[:, 18] - Jp[:, 16]).mean() + vec_angle(J[:, 19] - J[:, 17], Jp[:, 19] - Jp[:, 17]).mean()) / 2
                d_th = (vec_angle(J[:, 4] - J[:, 1], Jp[:, 4] - Jp[:, 1]).mean() + vec_angle(J[:, 5] - J[:, 2], Jp[:, 5] - Jp[:, 2]).mean()) / 2
                mpjpe = (Jp - J).norm(dim=-1)[:, [1, 2, 4, 5, 16, 17, 18, 19]].mean().item() * 100       # unsensed joints, cm (root-relative)
                res[k].append([float(e[js].mean()) for js in GROUPS.values()] + [d_arm.item(), d_th.item(), mpjpe])
        print(f'=== {title} (deg; last col = unsensed-joint position error cm) ===')
        print('variant'.ljust(10) + '  '.join(n[:12].rjust(12) for n in GROUPS) + '  upperarm-dir  thigh-dir   unsens-cm')
        for k in models:
            print(k.ljust(10) + '  '.join(f'{x:12.1f}' for x in np.average(res[k], axis=0, weights=wts)), flush=True)

    dip_eval('Test 1: DIP-IMU test split, real IMU, as given')
    dip_eval('Test 3a: DIP + random constant sensor offsets, sigma 10 deg', 10.0)
    dip_eval('Test 3b: DIP + random constant sensor offsets, sigma 20 deg', 20.0)
    # ---- test 4: own captures, as-given (uncalibrated)
    caps = {k: [] for k in models}
    for cap in sorted(Path('results/ours_multi_finetune/captures').iterdir(), key=lambda p: int(p.name.split('_')[0])):
        t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        gt = t['pose'].reshape(-1, 24, 3, 3)
        d = torch.load(cap / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
        T = min(len(d['acc'][0]), len(gt))
        for k, fn in models.items():
            Gp, _ = body.forward_kinematics(fn(d['acc'][0][:T], d['ori'][0][:T])[:T], torch.ones(10), tran=None, calc_mesh=False)
            Gq, _ = body.forward_kinematics(gt[:T], torch.ones(10), tran=None, calc_mesh=False)
            e = angle_between_rotmats(Gp.reshape(T, 24, 3, 3), Gq.reshape(T, 24, 3, 3)).mean(0).numpy()
            caps[k].append([float(e[js].mean()) for js in GROUPS.values()])
    print('=== Test 4: 10 own captures, as-given inputs (uncalibrated), mean over captures (deg) ===')
    print('variant'.ljust(10) + '  '.join(n[:12].rjust(12) for n in GROUPS))
    for k in models:
        print(k.ljust(10) + '  '.join(f'{x:12.1f}' for x in np.mean(caps[k], axis=0)))
    # ---- test 2: held-out AMASS events
    blob = torch.load('data/hold_events_heldout.pt', map_location='cpu', weights_only=False)
    rows = {k: [] for k in models}
    for sub, i, t0, end, T in blob['events']:
        h = blob['held'][sub]
        j = h['indices'].index(i)
        acc, ori, pose = h['acc'][j], h['ori'][j], h['pose'][j].reshape(-1, 24, 3, 3)
        T = min(len(pose), 3000)
        acc, ori, pose = acc[:T], ori[:T], pose[:T]
        G, _ = body.forward_kinematics(pose, torch.ones(10), tran=None, calc_mesh=False)
        G = G.reshape(-1, 24, 3, 3)
        end = min(end, T)
        for k, fn in models.items():
            Gp, _ = body.forward_kinematics(fn(acc, ori)[:T], torch.ones(10), tran=None, calc_mesh=False)
            terr = angle_between_rotmats(Gp.reshape(-1, 24, 3, 3)[:, 1], G[:, 1]).numpy()
            rows[k].append([terr[t0 + int(lo * 30):min(end, t0 + int(hi * 30))].mean()
                            if min(end, t0 + int(hi * 30)) - (t0 + int(lo * 30)) >= 10 else np.nan for lo, hi in BINS])
    print(f'\n=== Test 2: {len(blob["events"])} held-out AMASS stand->hold events, thigh error (deg), full-sequence inference ===')
    print('variant'.ljust(10) + '  '.join(f'{lo}-{hi if hi < 1000 else "end"}s'.rjust(8) for lo, hi in BINS))
    for k in models:
        print(k.ljust(10) + '  '.join(f'{x:8.1f}' for x in np.nanmean(np.array(rows[k]), axis=0)))


if __name__ == '__main__':
    main()
