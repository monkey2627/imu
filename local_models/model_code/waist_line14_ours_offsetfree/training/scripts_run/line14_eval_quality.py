"""Stratified evaluation of trained downstream FiLM models by calibration quality (E20).

usage: python scripts_run/line14_eval_quality.py --ckpt-glob 'checkpoints/line14/E20/estp3_oracle_film_s{seed}.pt' --geom estp3 \
           --est-file data_local/line14_est_r2_degr.pt --store data_local/line14_store_mount3 --out results/line14/E20q/estp3.json
Reports the validation 'all' joint-position error (identity-mean, cm) when every identity's estimate comes from a  good | bad | any  session.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from mobileposer.line9.models import Line9Joints, GeometryStats
from mobileposer.line14.geom import GeomProvider, scale_ident
from mobileposer.line14.models import FilmJoints
from mobileposer.line14.train_common import build_val, val_loader
from mobileposer.line9.evaluation import PositionEvaluationAccumulator


def run(model, val, device, seed, prov, sr):
    acc = PositionEvaluationAccumulator()
    with torch.no_grad():
        for b in val_loader(val):
            x = b['target_imu'].to(device); gt = b['target_joints']
            if sr:
                s = torch.tensor([scale_ident(i, *sr) for i in b['identity']], dtype=torch.float32)
                x = x.clone(); x[..., :15] = x[..., :15] * s.to(device)[:, None, None]; gt = gt * s[:, None, None, None]
            g = prov.batch(b['beta'], b['identity'], 'none', 0.0, fixed_seed=seed).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                p = model(x, b['target_lengths'].tolist(), g)
            acc.update(p.float().cpu().reshape(*p.shape[:2], 24, 3), gt, b['target_mask'], b['identity'])
    return acc.compute(n_bootstrap=100, seed=1)['aggregate']['all']['mean']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-glob', required=True); ap.add_argument('--geom', required=True); ap.add_argument('--est-file', required=True)
    ap.add_argument('--store', type=Path, default=Path('data_local/line14_store_mount3')); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--scale-range', default=''); ap.add_argument('--arch', default='film')
    a = ap.parse_args()
    device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    sr = tuple(float(x) for x in a.scale_range.split(',')) if a.scale_range else None
    prov = GeomProvider(a.geom, 3.0, a.est_file, sr)
    records, val = build_val(store=a.store)
    out = {}
    for seed in (42, 7, 3):
        st = torch.load(a.ckpt_glob.format(seed=seed), map_location=device, weights_only=False)
        stats = GeometryStats(mean=st['geometry_mean'], std=st['geometry_std'])
        model = (FilmJoints(prov.dim, stats) if a.arch == 'film' else Line9Joints(mode='oracle', geom_dim=prov.dim, geometry_stats=stats)).to(device)
        model.load_state_dict(st['model']); model.eval()
        for q in (None, 'good', 'bad'):
            prov.quality = q
            out.setdefault(str(q or 'any'), []).append(run(model, val, device, seed, prov, sr))
    res = {k: {'per_seed': v, 'mean': float(np.mean(v))} for k, v in out.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: round(v['mean'], 3) for k, v in res.items()}))


if __name__ == '__main__':
    main()
