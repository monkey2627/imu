"""Shared helpers for line-14 evaluation-only scripts (validation fold only)."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from mobileposer.line9.data import load_identity_records, PairedWindowDataset, collate_paired_windows, SubsetBatchSampler
from mobileposer.line9.store import SequenceStore
from mobileposer.line9.evaluation import PositionEvaluationAccumulator
from mobileposer.line14.layout import PNP5_DATA_ROOT, validate_pnp5_dataset_metadata

DATA_ROOT = PNP5_DATA_ROOT
AUDIT = Path('results/line9/data_audit.json')
STORE = Path('data/line9_store_shape')


def build_val(window: int = 125, smoke: bool = False, store: Path = STORE):
    validate_pnp5_dataset_metadata(DATA_ROOT)
    kw = dict(max_identities_per_fold=2, max_sequences_per_identity=2) if smoke else {}
    records = load_identity_records(DATA_ROOT, AUDIT, ('validation',), **kw)
    st = SequenceStore(store) if (store / 'index.json').is_file() else None
    val = PairedWindowDataset(records['validation'], window, max_windows=8 if smoke else None, file_cache_size=2, sequence_cache_size=64, store=st)
    return records, val


def val_loader(val, batch: int = 128, seed: int = 0):
    sampler = SubsetBatchSampler(val, batch, shuffle=False, drop_last=False, seed=seed)
    return DataLoader(val, batch_sampler=sampler, collate_fn=collate_paired_windows, num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate_pert(model, val, device, seed, prov, pert, bootstrap=1000):
    acc = PositionEvaluationAccumulator()
    for b in val_loader(val):
        x = b['target_imu'].to(device)
        g = prov.batch(b['beta'], b['identity'], pert[0], pert[1], fixed_seed=seed).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            p = model(x, b['target_lengths'].tolist(), g)
        acc.update(p.cpu().reshape(*p.shape[:2], 24, 3), b['target_joints'], b['target_mask'], b['identity'])
    return acc.compute(n_bootstrap=bootstrap, seed=20261013 + seed)
