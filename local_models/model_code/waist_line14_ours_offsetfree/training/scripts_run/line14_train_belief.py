"""Train the calibration-free causal belief-state model on PNP5 AMASS windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mobileposer.line9.data import load_identity_records
from mobileposer.line9.pose_store import PoseStore
from mobileposer.line14.belief import CausalBeliefFilter, belief_loss
from mobileposer.line14.layout import assert_pnp5_data_root, validate_pnp5_dataset_metadata
from scripts_run.line14_train_latent_adapter import TargetOnlyWindows, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/pnp5_60a/pnp5')
    ap.add_argument('--audit', default='results/line9/data_audit.json')
    ap.add_argument('--raw-root', default='/home/duanyuhan/dyh/motion/dataset/raw/AMASS')
    ap.add_argument('--imu-store', default='data/line9_store_calib6')
    ap.add_argument('--pose-store', default='data/line9_store_pose6d')
    ap.add_argument('--out', default='experiments/line14/belief_filter/checkpoints/belief_s42.pt')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--window', type=int, default=250)
    ap.add_argument('--max-windows', type=int, default=0)
    ap.add_argument('--max-batches', type=int, default=0)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.batch, a.max_windows, a.max_batches = 2, 4, 16, 2
    root = assert_pnp5_data_root(a.data_root)
    validate_pnp5_dataset_metadata(root)
    device = torch.device(a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    torch.manual_seed(14027)
    records = load_identity_records(root, a.audit, ('train', 'validation'), raw_root=a.raw_root,
                                    max_identities_per_fold=3 if a.smoke else None,
                                    max_sequences_per_identity=3 if a.smoke else None)
    store = PoseStore(a.imu_store, a.pose_store)
    calib_meta = Path(a.imu_store) / 'calib.json'
    if not calib_meta.is_file():
        raise FileNotFoundError(f"expected fixed identity-level mounting metadata: {calib_meta}")
    mount_spec = json.loads(calib_meta.read_text())
    if mount_spec.get('rot_deg_rms') != 5.0 or mount_spec.get('lever_cm') != 3.0:
        raise ValueError(f"unexpected mounting augmentation distribution: {mount_spec}")
    train = TargetOnlyWindows(records['train'], store, torch.empty(0), a.window, a.max_windows or (16 if a.smoke else None))
    val = TargetOnlyWindows(records['validation'], store, torch.empty(0), a.window, 8 if a.smoke else None)
    # The dataset uses FK only in __getitem__; this parent is replaced by the
    # canonical SMPL parent exposed by the store-compatible model at runtime.
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    parent = ParametricModel(paths.smpl_file, device=torch.device('cpu')).parent
    train.parent = parent; val.parent = parent
    tr = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=0, collate_fn=collate)
    va = DataLoader(val, batch_size=a.batch, shuffle=False, num_workers=0, collate_fn=collate)
    model = CausalBeliefFilter(hidden=a.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    def eval_loader(loader):
        model.eval(); total = n = 0
        with torch.no_grad():
            for bi, (imu, target, lengths) in enumerate(loader):
                if a.max_batches and bi >= a.max_batches: break
                imu, target = imu.to(device), target.to(device)
                out = model(imu)
                rows = torch.arange(len(lengths), device=device)
                end = lengths.to(device) - 1
                value = (out['pose'][rows, end] - target[rows, end]).square().mean()
                total += float(value) * len(lengths); n += len(lengths)
        return total / max(n, 1)

    history = []; best = float('inf')
    for epoch in range(a.epochs):
        model.train(); total = n = 0
        for bi, (imu, target, lengths) in enumerate(tr):
            if a.max_batches and bi >= a.max_batches: break
            imu, target = imu.to(device), target.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(imu)
            valid = torch.arange(imu.shape[1], device=device)[None] < lengths.to(device)[:, None]
            loss = belief_loss(out, imu, target, mask=valid)
            if not torch.isfinite(loss): raise FloatingPointError('non-finite belief loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()); n += 1
        val_loss = eval_loader(va)
        row = {'epoch': epoch + 1, 'train_loss': total / max(n, 1), 'val_pose_loss': val_loss}
        history.append(row); print(json.dumps(row), flush=True)
        if val_loss < best:
            best = val_loss
            out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'hidden': a.hidden, 'history': history,
                        'best_val_pose_loss': best, 'future_frames': 0,
                        'sensor_layout': ['lw', 'rw', 'ls', 'rs', 'pelvis'],
                        'calibration': 'none', 'status': 'smoke_only' if a.smoke else 'trained_not_benchmark_evaluated'}, out_path)
    print(json.dumps({'saved': a.out, 'best_val_pose_loss': best, 'history': history,
                      'future_frames': 0, 'calibration': 'none', 'mount_augmentation': mount_spec}))


if __name__ == '__main__': main()
