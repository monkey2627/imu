#!/usr/bin/env python3
"""Ours copy whose sensor orientations agree with the HumanPose labels' bone orientations.

Each capture's 5-IMU orientations differ from the label bone orientation by a
near-constant body-frame rotation per sensor (waist ~20 deg everywhere, forearms
4..106 deg varying per capture). That offset is estimated per capture and
removed (ori @ C, C = mean(R_sensor^T R_label)), so fine-tuning sees the AMASS
convention (sensor = bone) instead of memorising per-action offsets.

Fairness: training actions use the whole sequence; held-out actions only their
first 2 s, close to the single calibration available live. Labels are linked,
not copied or modified.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

JOINTS = [18, 19, 4, 5, 3]          # lw rw ls rs waist -> SMPL joint of the sensed bone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--holdout', default='9,10')
    ap.add_argument('--holdout-seconds', type=float, default=2.0)
    args = ap.parse_args()
    sys.path.insert(0, str(args.workspace))
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    body = ParametricModel(paths.smpl_file)
    holdout = set(args.holdout.split(','))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'kind': 'Ours with per-capture body-frame sensor offsets removed', 'source': str(args.source),
              'holdout': sorted(holdout), 'holdout_seconds': args.holdout_seconds, 'captures': {}}
    for capture in sorted((args.source / 'captures').iterdir(), key=lambda p: int(p.name)):
        inp = torch.load(capture / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
        tgt = torch.load(capture / 'targets_humanpose_calibrated_60.pt', map_location='cpu', weights_only=False)
        ori = inp['ori'][0].double()
        label = body.forward_kinematics(tgt['pose'].float())[0].double()
        frames = slice(0, int(args.holdout_seconds * 60)) if capture.name in holdout else slice(None)
        offsets = []
        for s, joint in enumerate(JOINTS):
            rel = ori[frames, s].transpose(-1, -2) @ label[frames, joint]
            offsets.append(Rotation.from_matrix(rel.numpy()).mean().as_matrix())
        offsets = torch.from_numpy(np.stack(offsets))
        fixed = ori.clone()
        fixed[:, :5] = ori[:, :5] @ offsets
        residual = [float(np.degrees(Rotation.from_matrix(
            (fixed[:, s].transpose(-1, -2) @ label[:, j]).numpy()).magnitude().mean())) for s, j in enumerate(JOINTS)]
        dest = args.output / 'captures' / capture.name
        (dest / 'inputs/wrists_shanks_waist').mkdir(parents=True)
        out = dict(inp)
        out['ori'] = [fixed.float()] + list(inp['ori'][1:])
        out['offset_removal'] = 'per-capture body-frame sensor offsets removed (see report.json)'
        torch.save(out, dest / 'inputs/wrists_shanks_waist/ours1.pt')
        (dest / 'targets_humanpose_calibrated_60.pt').symlink_to(capture / 'targets_humanpose_calibrated_60.pt')
        report['captures'][capture.name] = {
            'offset_deg': [float(np.degrees(Rotation.from_matrix(o.numpy()).magnitude())) for o in offsets],
            'residual_deg_whole_sequence': residual,
            'estimated_from': 'first %.1f s' % args.holdout_seconds if capture.name in holdout else 'whole sequence'}
        print(capture.name, 'offset', np.round(report['captures'][capture.name]['offset_deg'], 1),
              'residual', np.round(residual, 1))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
