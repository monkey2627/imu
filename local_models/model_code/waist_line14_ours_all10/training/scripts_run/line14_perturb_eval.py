"""E3 test-time perturbation of the geometry input for trained Oracle checkpoints (no retraining).

For each trained checkpoint (geom kind, seed) evaluate the validation set with:
  clean | zero-information (training-identity mean geometry) | shuffle (other identity's geometry)
  | indep noise {1,2,5,10,20}% | global (pure scale) noise {1,2,5,10,20}%
The noise is a deterministic function of (identity, seed, sigma), i.e. one fixed 'calibration estimate' per identity.
Writes results/line14/<out-tag>/<name>.json with per-perturbation metrics (same structure as line14_train).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from mobileposer.line9.models import Line9Joints, GeometryStats
from mobileposer.line14.geom import GeomProvider
from mobileposer.line14.train_common import build_val, evaluate_pert


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--ckpt', required=True, type=Path); a.add_argument('--seed', type=int, required=True)
    a.add_argument('--geom', default='r1'); a.add_argument('--gpu', type=int, default=0)
    a.add_argument('--out', required=True, type=Path); a.add_argument('--bootstrap', type=int, default=1000)
    a.add_argument('--sigmas', default='0.01,0.02,0.05,0.10,0.20'); a.add_argument('--add-sigmas-cm', default='0.5,1,2,3,5'); a.add_argument('--smoke', action='store_true')
    a = a.parse_args()
    device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    prov = GeomProvider(a.geom)
    records, val = build_val(smoke=a.smoke)
    prov.set_pool({r.identity: torch.tensor(r.beta) for r in records['validation']})
    st = torch.load(a.ckpt, map_location=device, weights_only=False)
    stats = GeometryStats(mean=st['geometry_mean'], std=st['geometry_std'])
    model = Line9Joints(mode='oracle', geom_dim=prov.dim, geometry_stats=stats).to(device)
    model.load_state_dict(st['model']); model.eval()
    results = {}
    perts = [('clean', ('none', 0.0)), ('shuffle', ('shuffle', 0.0))]
    for s in [float(x) for x in a.sigmas.split(',')]:
        perts += [(f'indep_{int(round(s*100))}', ('indep', s)), (f'global_{int(round(s*100))}', ('global', s))]
    for s in [float(x) for x in a.add_sigmas_cm.split(',') if x]:
        perts += [(f'add_{s:g}cm', ('add', s))]
    for name, pert in perts:
        results[name] = evaluate_pert(model, val, device, a.seed, prov, pert, a.bootstrap if not a.smoke else 10)
        print(name, 'all=%.3f' % results[name]['aggregate']['all']['mean'], flush=True)
    # zero-information: the training-identity mean geometry == normalised vector 0
    zero_vec = model.geometry_mean.detach().cpu()
    orig_batch = prov.batch
    prov.batch = lambda betas, identities=None, mode='none', sigma=0.0, fixed_seed=None: zero_vec.unsqueeze(0).expand(len(betas), -1).clone()
    results['mean_geometry'] = evaluate_pert(model, val, device, a.seed, prov, ('none', 0.0), a.bootstrap if not a.smoke else 10)
    prov.batch = orig_batch
    print('mean_geometry', 'all=%.3f' % results['mean_geometry']['aggregate']['all']['mean'], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'ckpt': str(a.ckpt), 'seed': a.seed, 'geom': a.geom, 'status': 'smoke_only' if a.smoke else 'formal_validation_only',
                                 'test_accessed': False, 'results': results}, indent=1))


if __name__ == '__main__':
    main()
