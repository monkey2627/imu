"""Legacy Line 14 geometry-conditioned Joints diagnostic trainer.

NON-CAUSAL: ``Line9Joints`` uses a bidirectional LSTM. Its outputs are not
eligible as the strict-real-time line14 primary method or as a zero-lookahead
result. The main causal model needs its own explicitly unidirectional trainer.

Same protocol as line14_envB.py / line 9 (AdamW 1e-3 wd 0.01, 60 epochs, batch 128, window 125, identity-level
train/validation split, store data/line9_store_shape), but with a selectable geometry input and optional training noise:

  --geom {r1,height,arms,legs}          geometry kind (see mobileposer/line14/geom.py)
  --noise-train SIGMA --noise-mode {indep,global}   fresh relative noise on the geometry at every training step
  --tag NAME                            output sub-directory name (results/line14/<tag>/, checkpoints/line14/<tag>/)
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from mobileposer.line9.data import (load_identity_records, PairedWindowDataset,
    collate_paired_windows, SubsetBatchSampler)
from mobileposer.line9.store import SequenceStore
from mobileposer.line9.models import build_fair_models, fit_geometry_stats, parameter_count_report
from mobileposer.line9.evaluation import PositionEvaluationAccumulator
from mobileposer.line14.geom import GeomProvider, scale_ident
from mobileposer.line14.layout import (PNP5_DATA_ROOT, assert_pnp5_data_root,
                                       pnp5_layout_metadata, validate_pnp5_dataset_metadata,
                                       validate_pnp5_surface_config)


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def geometry_rows(prov, dataset, records):
    rows = []; seen = set()
    for rec in records:
        if rec.identity not in seen:
            beta, _ = dataset._get_geometry(rec)
            rows.append(prov.base(beta, rec.identity)); seen.add(rec.identity)
    return torch.stack(rows)


def loader(dataset, batch, seed, shuffle):
    sampler = SubsetBatchSampler(dataset, batch, shuffle=shuffle, drop_last=False, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_paired_windows, num_workers=0, pin_memory=True), sampler


def _scales(b, sr, device):
    return torch.tensor([scale_ident(i, *sr) for i in b['identity']], dtype=torch.float32, device=device)


def evaluate(model, dl, device, seed, prov, bootstrap=10000, pert=("none", 0.0), sr=None):
    model.eval(); acc = PositionEvaluationAccumulator()
    with torch.no_grad():
        for b in dl:
            x = b['target_imu'].to(device)
            if sr:
                s = _scales(b, sr, device); x = x.clone(); x[..., :15] = x[..., :15] * s[:, None, None]
                b = dict(b); b['target_joints'] = b['target_joints'] * s.cpu()[:, None, None, None]
            geom = prov.batch(b['beta'], b['identity'], pert[0], pert[1], fixed_seed=seed).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                p = model(x, b['target_lengths'].tolist(), geom)
            acc.update(p.cpu().reshape(*p.shape[:2], 24, 3), b['target_joints'], b['target_mask'], b['identity'])
    return acc.compute(n_bootstrap=bootstrap, seed=20261013 + seed)


def build_data(a):
    records = load_identity_records(a.data_root, a.audit, ('train', 'validation'),
                                    max_identities_per_fold=a.max_identities, max_sequences_per_identity=a.max_sequences)
    store = SequenceStore(a.store) if (a.store / 'index.json').is_file() else None
    train = PairedWindowDataset(records['train'], a.window, max_windows=a.max_windows, file_cache_size=2, sequence_cache_size=64, store=store)
    val = PairedWindowDataset(records['validation'], a.window, max_windows=a.max_windows, file_cache_size=2, sequence_cache_size=64, store=store)
    return records, train, val


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--mode', choices=['b0', 'b1', 'oracle'], required=True); a.add_argument('--seed', type=int, required=True)
    a.add_argument('--gpu', type=int, default=0); a.add_argument('--geom', default='r1', choices=['r1', 'height', 'arms', 'legs', 'r1m', 'r2', 'r2m', 'est', 'est_sig', 'r2scale', 'est_scale', 'pca1', 'pca2', 'pca3', 'pca4', 'pca6', 'estp1', 'estp2', 'estp3', 'estp4', 'estp6', 'estp1s', 'estp3s', 'estp1g', 'estp3g', 'hestp3', 'hestp6', 'hrestp3', 'hrestp6'])
    a.add_argument('--noise-train', type=float, default=0.0); a.add_argument('--noise-mode', default='indep', choices=['indep', 'global', 'add'])
    a.add_argument('--tag', default='E2'); a.add_argument('--scale-range', default=''); a.add_argument('--arch', default='add', choices=['add', 'film']); a.add_argument('--lever-cm', type=float, default=3.0); a.add_argument('--est-file', default=None)
    a.add_argument('--epochs', type=int, default=60); a.add_argument('--batch', type=int, default=128); a.add_argument('--lr', type=float, default=1e-3)
    a.add_argument('--weight-decay', type=float, default=.01); a.add_argument('--window', type=int, default=125)
    a.add_argument('--data-root', type=Path, default=PNP5_DATA_ROOT); a.add_argument('--audit', type=Path, default=Path('results/line9/data_audit.json'))
    a.add_argument('--store', type=Path, default=Path('data/line9_store_shape'))
    a.add_argument('--max-identities', type=int); a.add_argument('--max-sequences', type=int); a.add_argument('--max-windows', type=int)
    a.add_argument('--smoke', action='store_true'); a = a.parse_args()
    if a.smoke: a.epochs = 1; a.max_identities = a.max_identities or 2; a.max_sequences = a.max_sequences or 2; a.max_windows = a.max_windows or 8; a.batch = min(a.batch, 4)
    a.data_root = assert_pnp5_data_root(a.data_root)
    validate_pnp5_dataset_metadata(a.data_root)
    validate_pnp5_surface_config()
    print('CAUSALITY WARNING: this geometry diagnostic uses bidirectional Line9Joints and is not a line14 primary result.', flush=True)
    seed_all(a.seed); device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    sr = tuple(float(x) for x in a.scale_range.split(',')) if a.scale_range else None
    prov = GeomProvider(a.geom, a.lever_cm, a.est_file, sr)
    records, train, val = build_data(a)
    print('geom', a.geom, 'dim', prov.dim, 'noise_train', a.noise_train, a.noise_mode, flush=True)
    stats = fit_geometry_stats(geometry_rows(prov, train, records['train']))
    models = build_fair_models(stats); counts = parameter_count_report(models); model = models[a.mode].to(device); del models
    if a.arch == 'film':
        from mobileposer.line14.models import FilmJoints
        model = FilmJoints(prov.dim, stats).to(device); counts = parameter_count_report({'film': model})
    td, ts = loader(train, a.batch, a.seed, True); vd, vs = loader(val, a.batch, a.seed, False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay); scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    name = f"{a.geom}_{a.mode}" + ("_film" if a.arch == 'film' else "") + ((f"_add{a.noise_train:g}cm" if a.noise_mode == 'add' else f"_n{int(round(a.noise_train*100))}{a.noise_mode[0]}") if a.noise_train > 0 else "") + f"_s{a.seed}"
    out = Path('results/line14') / a.tag / f'{name}.json'; ck = Path('checkpoints/line14') / a.tag / f'{name}.pt'
    out.parent.mkdir(parents=True, exist_ok=True); ck.parent.mkdir(parents=True, exist_ok=True)
    best = float('inf'); history = []; started = time.time()
    tnoise = (a.noise_mode, a.noise_train) if a.noise_train > 0 else ('none', 0.0)
    for epoch in range(a.epochs):
        ts.set_epoch(epoch); model.train(); total = n = 0
        for b in td:
            x = b['target_imu'].to(device); y = b['target_joints'].to(device).flatten(2); mask = b['target_mask'].to(device)
            if sr:
                s = _scales(b, sr, device); x = x.clone(); x[..., :15] = x[..., :15] * s[:, None, None]; y = y * s[:, None, None]
            geom = prov.batch(b['beta'], b['identity'], tnoise[0], tnoise[1], fixed_seed=None).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                p = model(x, b['target_lengths'].tolist(), geom); mse = ((p - y).square() * mask[:, :, None]).sum() / (mask.sum() * 72)
                acc = p[:, 2:] + p[:, :-2] - 2 * p[:, 1:-1]; tm = mask[:, 2:] & mask[:, :-2] & mask[:, 1:-1]
                temporal = (acc.abs() * tm[:, :, None]).sum() / tm.sum().clamp_min(1)
                loss = mse + 1e-5 * temporal
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total += float(loss) * int(mask.sum()); n += int(mask.sum())
        model.eval(); ve = vn = 0
        with torch.no_grad():
            for b in vd:
                x = b['target_imu'].to(device); y = b['target_joints'].to(device).flatten(2); m = b['target_mask'].to(device)
                if sr:
                    s = _scales(b, sr, device); x = x.clone(); x[..., :15] = x[..., :15] * s[:, None, None]; y = y * s[:, None, None]
                g = prov.batch(b['beta'], b['identity']).to(device)   # clean geometry for model selection
                p = model(x, b['target_lengths'].tolist(), g); e = (p.reshape(*p.shape[:2], 24, 3) - y.reshape(*y.shape[:2], 24, 3)).norm(dim=-1)
                ve += float((e * m[:, :, None]).sum()); vn += int(m.sum()) * 24
        score = 100 * ve / vn; history.append({'epoch': epoch + 1, 'train_loss': total / max(n, 1), 'validation_all_cm_pooled': score})
        if score < best:
            best = score; torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'score': score, 'geometry_mean': stats.mean, 'geometry_std': stats.std,
                                      'geom': a.geom, 'mode': a.mode}, ck)
        print(json.dumps({'mode': a.mode, 'geom': a.geom, 'seed': a.seed, **history[-1]}), flush=True)
    state = torch.load(ck, map_location=device, weights_only=False); model.load_state_dict(state['model'])
    metrics = evaluate(model, vd, device, a.seed, prov, 10 if a.smoke else 10000, sr=sr)
    result = {'status': 'smoke_only' if a.smoke else 'formal_validation_only', 'mode': a.mode, 'geom': a.geom, 'seed': a.seed,
              'noise_train': {'mode': a.noise_mode, 'sigma': a.noise_train}, 'scale_range': sr,
              'protocol': {'sensor_layout': pnp5_layout_metadata(), 'causality': 'bidirectional_noncausal_diagnostic_only', 'primary_line14_result': False, 'epochs': a.epochs, 'batch': a.batch, 'window': a.window, 'optimizer': 'AdamW', 'lr': a.lr, 'weight_decay': a.weight_decay,
                           'train_identities': len({r.identity for r in records['train']}), 'validation_identities': len({r.identity for r in records['validation']}),
                           'train_sequences': len(records['train']), 'validation_sequences': len(records['validation']),
                           'target_window_digest_train': train.target_window_digest(), 'target_window_digest_validation': val.target_window_digest(), 'test_accessed': False},
              'parameter_counts': counts, 'best_epoch': state['epoch'], 'history': history, 'metrics': metrics, 'elapsed_seconds': time.time() - started}
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2)); os.replace(tmp, out)


if __name__ == '__main__':
    main()
