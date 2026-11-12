#!/usr/bin/env python3
"""AMASS-format synthetic 5-IMU data from the Ours HumanPose labels, added to the replay pool.

The Ours labels (SMPL pose, translation, shape) are re-synthesised with exactly the
AMASS surface pipeline (mobileposer.no_head_layouts._synthesize, same attachment
config), so the inputs are offset-free by construction and cover the lying /
kneeling motions AMASS lacks. Each label sequence is cut into overlapping 5 s
chunks and synthesised for several perturbed body shapes, so the chunks make up
a noticeable share of the replay samples. The replay directory links the
original AMASS files (untouched) and adds OursSynth_<actions>.pt.

Run with MOBILEPOSER_FPS=60 from anywhere; --workspace is the pnp workspace.
"""
import argparse
import os
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--ours', type=Path, required=True, help='OursHumanPose_* dataset (labels are read from it)')
    ap.add_argument('--amass', type=Path, required=True, help='existing AMASS surface replay directory')
    ap.add_argument('--output', type=Path, required=True, help='new replay directory')
    ap.add_argument('--actions', default='1,2,3,4,5,6,7,8')
    ap.add_argument('--shapes', type=int, default=10)
    ap.add_argument('--shape-sigma', type=float, default=0.8)
    ap.add_argument('--chunk', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    assert os.environ.get('MOBILEPOSER_FPS') == '60', 'set MOBILEPOSER_FPS=60'
    sys.path.insert(0, str(args.workspace))
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    from mobileposer.no_head_layouts import LAYOUTS, _foot_ground_probs, _synthesize, synthesis_metadata

    config = args.workspace / 'mobileposer/configs/no_head_surface_imu.json'
    synthesis = synthesis_metadata('surface', config)
    synthesis.update({'source': 'Ours HumanPose labels (pseudo-labels) resynthesised', 'actions': args.actions,
                      'shapes': args.shapes, 'shape_sigma': args.shape_sigma, 'chunk_frames': args.chunk})
    body = ParametricModel(paths.smpl_file)
    generator = torch.Generator().manual_seed(args.seed)
    layout = 'wrists_shanks_waist'
    out = {k: [] for k in ('joint', 'pose', 'shape', 'tran', 'acc', 'ori', 'contact')}
    for action in args.actions.split(','):
        target = torch.load(args.ours / 'captures' / action / 'targets_humanpose_calibrated_60.pt',
                            map_location='cpu', weights_only=False)
        pose, tran, base_shape = target['pose'].float(), target['tran'].float(), target['shape'].float()
        starts = list(range(0, max(1, len(pose) - args.chunk + 1), args.chunk // 2))
        for k in range(args.shapes):
            shape = base_shape if k == 0 else base_shape + args.shape_sigma * torch.randn(10, generator=generator)
            for start in starts:
                p, t = pose[start:start + args.chunk], tran[start:start + args.chunk]
                if len(p) < 240:
                    continue
                with torch.no_grad():
                    ori, joint, acc = _synthesize(body, p, shape, t, layout, synthesis, 128)
                out['pose'].append(p.clone()); out['shape'].append(shape.clone()); out['tran'].append(t.clone())
                out['joint'].append(joint[:, :24].contiguous()); out['acc'].append(acc); out['ori'].append(ori)
                out['contact'].append(_foot_ground_probs(joint))
        print(f'action {action}: {len(starts)} chunks x {args.shapes} shapes', flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    for source in sorted(args.amass.glob('*.pt')):
        (args.output / source.name).symlink_to(source.resolve())
    name = 'OursSynth_' + args.actions.replace(',', '-') + '.pt'
    torch.save({'synthesis': synthesis, 'layout': layout, 'layout_labels': LAYOUTS[layout]['labels'],
                'layout_joints': LAYOUTS[layout]['joints'], **out}, args.output / name)
    print(f'{len(out["pose"])} synthetic sequences -> {args.output / name}')


if __name__ == '__main__':
    main()
