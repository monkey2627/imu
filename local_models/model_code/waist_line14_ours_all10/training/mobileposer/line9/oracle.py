"""Identity-paired AMASS episodes and Oracle-geometry baselines.

This module intentionally does not import or modify the public Joints model.
It consumes the immutable pnp5_60a cache and the line9 audit manifest.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _beta_key(x: torch.Tensor) -> str:
    a = np.round(x.detach().cpu().numpy()[:10].astype(np.float64), 4)
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


def load_manifest(audit: Path) -> dict:
    with audit.open() as f:
        return json.load(f)


def build_records(root: Path, audit: Path, folds: List[str], max_per_id: int = 0) -> Dict[str, List[dict]]:
    """Map audit's sorted raw sequence order to processed cache rows."""
    report = load_manifest(audit)
    wanted = {x for fold in folds for x in report['proposed_identity_split'][fold]['items']}
    excluded = set(report['protocol'].get('excluded_subsets', []))
    records = {k: [] for k in folds}
    id_fold = {x: fold for fold in folds for x in report['proposed_identity_split'][fold]['items']}
    for subset_file in sorted(root.glob('*.pt')):
        subset = subset_file.stem
        if subset in excluded or subset == 'TotalCapture': continue
        rows = torch.load(subset_file, map_location='cpu', weights_only=False)
        # Cache generation preserves the audit's valid raw path order. Advance the
        # cache cursor for every valid sequence, including identities outside this fold.
        raw_root = Path(report.get('_raw_amass_root', 'data/raw/AMASS'))
        paths = sorted(raw_root.joinpath(subset).glob('*/*_poses.npz'))
        row = 0
        for p in paths:
            with np.load(p) as z:
                fps = int(z['mocap_framerate']); frames = int(len(z['poses']))
                beta = torch.from_numpy(np.asarray(z['betas'][:10], dtype=np.float32))
            if fps < 60 or int(np.floor(frames / float(fps) * 60)) <= 12:
                continue
            ident = f'{subset}/{p.parent.name}'
            if ident in wanted and row < len(rows['shape']):
                rec = {'subset': subset, 'identity': ident, 'row': row,
                       'beta': beta, 'cache': subset_file}
                records[id_fold[ident]].append(rec)
            row += 1
    for fold in records:
        by_id = {}
        for r in records[fold]: by_id.setdefault(r['identity'], []).append(r)
        if max_per_id:
            for ident in by_id: by_id[ident] = by_id[ident][:max_per_id]
        records[fold] = [r for ident in sorted(by_id) for r in by_id[ident]]
    return records


class EpisodeDataset(Dataset):
    def __init__(self, records: List[dict], window: int = 125, seed: int = 0):
        self.items = []
        rng = random.Random(seed)
        grouped = {}
        for r in records: grouped.setdefault(r['identity'], []).append(r)
        for ident, seqs in sorted(grouped.items()):
            for target in seqs:
                calib = rng.choice([x for x in seqs if x is not target])
                self.items.append((calib, target))
        self.window = window
        self._cache = {}
        self._files = OrderedDict()

    def _get(self, rec):
        key = (str(rec['cache']), rec['row'])
        if key not in self._cache:
            path = str(rec['cache'])
            if path not in self._files:
                self._files[path] = torch.load(rec['cache'], map_location='cpu', weights_only=False)
                while len(self._files) > 3: self._files.popitem(last=False)
            else:
                self._files.move_to_end(path)
            d = self._files[path]
            acc = d['acc'][rec['row']].float().reshape(len(d['acc'][rec['row']]), -1)[:, :15] / 30.0
            ori = d['ori'][rec['row']].float()
            if ori.ndim == 4: ori = ori.reshape(len(ori), -1)
            self._cache[key] = (acc, ori[:, :45], d['joint'][rec['row']].float(), rec['beta'])
        return self._cache[key]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        c, t = self.items[i]
        ca, co, _, beta = self._get(c); ta, to, y, _ = self._get(t)
        cx = torch.cat((ca, co.reshape(len(co), -1)), -1)
        tx = torch.cat((ta, to.reshape(len(to), -1)), -1)
        def crop(x):
            if len(x) <= self.window: return x
            return x[:self.window]
        return crop(cx), crop(tx), crop(y), beta, t['identity']


def collate(batch):
    c, x, y, b, ids = zip(*batch)
    def pad(xs):
        n = max(len(v) for v in xs); out = xs[0].new_zeros((len(xs), n) + tuple(xs[0].shape[1:]))
        m = torch.zeros(len(xs), n, dtype=torch.bool)
        for i, v in enumerate(xs): out[i, :len(v)] = v; m[i, :len(v)] = 1
        return out, m
    return (*pad(c), *pad(x), *pad(y), torch.stack(b), ids)


class OracleJoints(nn.Module):
    def __init__(self, mode='b0', hidden=128, geom_dim=10):
        super().__init__(); self.mode = mode
        self.encoder = nn.GRU(60, hidden, batch_first=True, bidirectional=True)
        extra = 0 if mode == 'b0' else geom_dim
        self.constant = nn.Parameter(torch.zeros(geom_dim)) if mode == 'b1' else None
        self.head = nn.Sequential(nn.Linear(hidden * 2 + extra, hidden), nn.ReLU(), nn.Linear(hidden, 72))

    def forward(self, calib, target, beta):
        _, hc = self.encoder(calib); z = torch.cat((hc[-2], hc[-1]), -1)
        if self.mode == 'oracle':
            g = beta[:, :10]
        elif self.mode == 'b1':
            g = self.constant[None].expand(len(beta), -1)
        else: g = None
        h, _ = self.encoder(target)
        if g is not None: h = torch.cat((h, g[:, None].expand(-1, h.shape[1], -1)), -1)
        return self.head(h)
