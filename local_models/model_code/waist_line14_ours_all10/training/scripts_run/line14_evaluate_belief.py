"""Strict-causal PNP5 evaluation for the line14 belief-state checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mobileposer.line14.belief import CausalBeliefFilter
from mobileposer.articulate import math as M
from mobileposer.evaluate_standard import StdEvaluator, prepare_imu, sequences
from mobileposer.models.poser import Poser
from mobileposer.evaluate_standard import resolve_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='experiments/line14/belief_filter/checkpoints/belief_s42_full.pt')
    ap.add_argument('--out', default='results/standard/line14_belief_s42_causal40.json')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--decoder-poser', default='checkpoints/pnp5_60a_s42_poser')
    a = ap.parse_args()
    device = torch.device(a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(a.checkpoint, map_location='cpu')
    model = CausalBeliefFilter(hidden=int(ckpt.get('hidden', 256))).to(device).eval()
    model.load_state_dict(ckpt['model'])
    # Reuse only the canonical SMPL global-to-local decoder and ignored-joint
    # convention from the baseline; the belief network supplies all 16 global
    # reduced rotations itself.
    decoder = Poser.load_from_checkpoint(str(resolve_checkpoint(a.decoder_poser)), map_location=device).to(device).eval()
    ev = StdEvaluator(device)
    result = {
        'name': 'line14_belief_s42_causal40',
        'checkpoint': str(Path(a.checkpoint)),
        'sensor_layout': ['lw', 'rw', 'ls', 'rs', 'pelvis'],
        'sensor_count': 5,
        'temporal_protocol': {'model_fps': 30, 'future_frames': 0, 'lookahead_ms': 0.0},
        'training': {'epochs': len(ckpt.get('history', [])), 'best_val_pose_loss': ckpt.get('best_val_pose_loss'),
                     'calibration': ckpt.get('calibration', 'none')},
    }
    with torch.no_grad():
        # The GRU is one-way. Run each sequence as one stream and never append
        # future frames to a target prefix.
        for dataset in ('dipimu', 'totalcapture_dipcalib', 'totalcapture_officalib'):
            errs = []
            for name, acc, ori, pose_t in sequences(dataset, ev.art):
                imu = prepare_imu(acc, ori).unsqueeze(0).to(device)
                pred6 = model(imu)['pose'][0]
                pred24 = decoder._reduced_global_to_full(pred6).to(device)
                errs.append(ev(pred24.cpu(), pose_t))
            result[dataset] = torch.stack(errs).mean(0).tolist()
            result.setdefault('sample_counts', {})[dataset] = len(errs)
            print(dataset, len(errs), result[dataset], flush=True)
        gen = torch.Generator().manual_seed(14027)
        x = torch.randn(1, 12, 60, generator=gen).to(device)
        base = model(x[:, :6])['pose']
        changed = x.clone(); changed[:, 6:] = torch.randn(changed[:, 6:].shape, generator=gen) * 100
        changed = changed.to(device)
        perturbed = model(changed[:, :6])['pose']
        result['future_invariance_max_abs_error'] = float((base - perturbed).abs().amax())
        if not torch.isfinite(pred24).all() or result['future_invariance_max_abs_error'] != 0.0:
            raise RuntimeError('belief checkpoint failed finite/future-invariance audit')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
