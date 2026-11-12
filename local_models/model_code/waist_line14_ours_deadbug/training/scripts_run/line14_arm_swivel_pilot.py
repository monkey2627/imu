"""Matched causal AMASS screening run for XYZ vs arm-nullspace Joints.

This is a small engineering/selection pilot, not the final line14 benchmark.
It uses the immutable line9 identity split, the same paired target windows,
and an identical optimizer/update budget for both model variants.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
# CUBLAS reads this when CUDA initializes; set it before importing torch.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import DataLoader


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(dataset, batch_size: int, seed: int, shuffle: bool):
    from mobileposer.line9.data import SubsetBatchSampler, collate_paired_windows

    sampler = SubsetBatchSampler(dataset, batch_size, shuffle=shuffle, drop_last=False, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_paired_windows, num_workers=0), sampler


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return (value * expanded).sum() / expanded.expand_as(value).sum().clamp_min(1)


def _loss(model, x, target, mask, lengths):
    prediction, aux = model._predict(x, lengths)
    pred = prediction.reshape(*prediction.shape[:2], 24, 3)
    mse = _masked_mean((pred - target).square(), mask)
    if prediction.shape[1] >= 3:
        accel = pred[:, 2:] + pred[:, :-2] - 2 * pred[:, 1:-1]
        tmask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        temporal = _masked_mean(accel.abs(), tmask)
    else:
        temporal = mse.new_zeros(())
    loss = mse + model.t_weight * temporal
    if aux is not None:
        valid = mask
        loss = loss + model.arm_reach_weight * _masked_mean(aux['reach_correction'].square(), valid)
        q = torch.nn.functional.normalize(aux['swivel'], dim=-1, eps=1e-6)
        if q.shape[1] > 1:
            continuity = 1.0 - (q[:, 1:] * q[:, :-1]).sum(-1).clamp(-1.0, 1.0)
            loss = loss + model.arm_swivel_weight * _masked_mean(continuity, valid[:, 1:] & valid[:, :-1])
        loss = loss + model.arm_orientation_weight * _masked_mean(aux['orientation_error'], valid)
        loss = loss + model.arm_residual_weight * _masked_mean(aux['residual'].square(), valid)
        loss = loss + model.arm_bone_weight * _masked_mean(aux['bone_error'].square(), valid)
    return loss


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    joint_sum = torch.zeros(24, dtype=torch.float64)
    joint_count = torch.zeros(24, dtype=torch.float64)
    swivel_step_deg = []
    swivel_lr_sin = [[], []]
    for batch in loader:
        x = batch['target_imu'].to(device)
        lengths = batch['target_lengths'].tolist()
        target = batch['target_joints'].to(device)
        mask = batch['target_mask'].to(device)
        pred, aux = model._predict(x, lengths)
        error = (pred.reshape(*pred.shape[:2], 24, 3) - target).norm(dim=-1)
        joint_sum += (error.double() * mask[:, :, None]).sum((0, 1)).cpu()
        joint_count += (mask.sum().double() * torch.ones(24, device=device)).cpu()
        if aux is not None:
            q = torch.nn.functional.normalize(aux['swivel'], dim=-1, eps=1e-6)
            valid = mask
            if q.shape[1] > 1:
                dot = (q[:, 1:] * q[:, :-1]).sum(-1).clamp(-1.0, 1.0)
                pair = valid[:, 1:] & valid[:, :-1]
                swivel_step_deg.extend((torch.acos(dot[pair]) * 180 / torch.pi).cpu().tolist())
            for side in range(2):
                swivel_lr_sin[side].extend(q[..., side, 0][valid].cpu().tolist())
    per_joint_cm = (100 * joint_sum / joint_count.clamp_min(1)).tolist()
    groups = {
        'shoulder': [16, 17],
        'elbow': [18, 19],
        'wrist': [20, 21],
    }
    metrics = {
        'per_joint_position_error_cm': per_joint_cm,
        'shoulder_cm': float(np.mean([per_joint_cm[j] for j in groups['shoulder']])),
        'elbow_cm': float(np.mean([per_joint_cm[j] for j in groups['elbow']])),
        'wrist_cm': float(np.mean([per_joint_cm[j] for j in groups['wrist']])),
        'all_joint_cm': float(np.mean(per_joint_cm)),
    }
    if swivel_step_deg:
        metrics['swivel_step_deg_median'] = float(np.median(swivel_step_deg))
        metrics['swivel_step_deg_mean'] = float(np.mean(swivel_step_deg))
        left, right = np.asarray(swivel_lr_sin[0]), np.asarray(swivel_lr_sin[1])
        if left.size > 3 and left.size == right.size and left.std() > 1e-8 and right.std() > 1e-8:
            metrics['left_right_sin_correlation'] = float(np.corrcoef(left, right)[0, 1])
        else:
            metrics['left_right_sin_correlation'] = None
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--window', type=int, default=125)
    parser.add_argument('--max-identities', type=int, default=2)
    parser.add_argument('--max-sequences', type=int, default=2)
    parser.add_argument('--max-windows', type=int, default=8)
    parser.add_argument('--data-root', type=Path, default=Path('data/pnp5_60a/pnp5'))
    parser.add_argument('--audit', type=Path, default=Path('results/line9/data_audit.json'))
    parser.add_argument('--raw-root', type=Path, default=Path('/home/duanyuhan/dyh/motion/MotionRe/base_mobileposer/data/raw/AMASS'))
    parser.add_argument('--store', type=Path, default=Path('data/line9_store_shape'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    os.environ['MOBILEPOSER_CAUSAL'] = '1'
    os.environ['MOBILEPOSER_FPS'] = '60'
    os.environ['MOBILEPOSER_NSENS'] = '5'
    os.environ['MOBILEPOSER_TRAIN_COMBOS'] = 'all_5imu'
    os.environ['MOBILEPOSER_ARCH'] = 'lstm'
    os.environ.pop('MOBILEPOSER_CTX', None)
    os.environ.pop('MOBILEPOSER_DOMAIN', None)

    from mobileposer.config import model_config
    from mobileposer.line9.data import load_identity_records, PairedWindowDataset
    from mobileposer.line9.store import SequenceStore
    from mobileposer.models.joints import Joints

    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model_config.device = device
    records = load_identity_records(
        args.data_root, args.audit, ('train', 'validation'), raw_root=args.raw_root,
        verify=True, max_identities_per_fold=args.max_identities,
        max_sequences_per_identity=args.max_sequences,
    )
    store = SequenceStore(args.store) if (args.store / 'index.json').is_file() else None
    train = PairedWindowDataset(records['train'], args.window, include_tail=False,
                                max_windows=args.max_windows, store=store)
    val = PairedWindowDataset(records['validation'], args.window, include_tail=False,
                              max_windows=args.max_windows, store=store)
    if not train.target_window_keys or not val.target_window_keys:
        raise RuntimeError('empty fixed-split pilot dataset')
    train_loader, train_sampler = _loader(train, args.batch, args.seed, True)
    val_loader, _ = _loader(val, args.batch, args.seed, False)
    results = {
        'status': 'screening_only',
        'seed': args.seed,
        'future_frames': 0,
        'protocol': {
            'epochs': args.epochs, 'batch': args.batch, 'window': args.window,
            'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 0.01,
            'train_identities': sorted({r.identity for r in records['train']}),
            'validation_identities': sorted({r.identity for r in records['validation']}),
            'train_window_digest': train.target_window_digest(),
            'validation_window_digest': val.target_window_digest(),
            'train_windows': len(train), 'validation_windows': len(val),
            'test_accessed': False,
        },
        'models': {},
    }

    for variant in ('xyz', 'arm_nullspace'):
        seed_all(args.seed)
        if variant == 'arm_nullspace':
            os.environ['MOBILEPOSER_ARM_NULLSPACE'] = '1'
        else:
            os.environ.pop('MOBILEPOSER_ARM_NULLSPACE', None)
        model = Joints().to(device)
        if model.joints.rnn.bidirectional:
            raise AssertionError('pilot model unexpectedly uses future frames')
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        history = []
        started = time.time()
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            model.train(); total = 0.0; steps = 0
            for batch in train_loader:
                x = batch['target_imu'].to(device)
                y = batch['target_joints'].to(device)
                mask = batch['target_mask'].to(device)
                lengths = batch['target_lengths'].tolist()
                opt.zero_grad(set_to_none=True)
                loss = _loss(model, x, y, mask, lengths)
                loss.backward(); opt.step()
                total += float(loss.detach()); steps += 1
            history.append({'epoch': epoch + 1, 'train_loss_mean_batch': total / max(steps, 1), 'steps': steps})
        results['models'][variant] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'history': history,
            'validation': _evaluate(model, val_loader, device),
            'elapsed_seconds': time.time() - started,
        }
        del model, opt

    results['matched_window_digests'] = (
        results['protocol']['train_window_digest'], results['protocol']['validation_window_digest']
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + '.tmp')
    tmp.write_text(json.dumps(results, indent=2))
    os.replace(tmp, args.out)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
