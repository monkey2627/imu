"""Train the calibration-free causal latent sensor-frame adapter.

This is an endpoint-supervised pilot.  The pretrained PNP5 Joints and Poser
backbones are frozen; only the zero-initialized causal adapter is optimized.
Each prediction is queried at the end of an observed prefix, so the frozen
bidirectional baseline RNNs never see frames later than the prediction time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mobileposer.articulate import math as M
from mobileposer.config import joint_set
from mobileposer.line9.data import load_identity_records
from mobileposer.line9.pose_store import PoseStore
from mobileposer.models.joints import Joints
from mobileposer.models.poser import Poser
from mobileposer.line14.adaptive import CausalLatentSensorAdapter
from mobileposer.line14.layout import assert_pnp5_data_root, validate_pnp5_dataset_metadata


class TargetOnlyWindows(Dataset):
    """Target motion windows only; no calibration clip is selected or loaded."""

    def __init__(self, records, store: PoseStore, parent: torch.Tensor, window: int, max_windows: int | None = None):
        self.store = store
        self.parent = parent
        self.rows = []
        for record in records:
            for start in range(0, record.frames, window):
                stop = min(start + window, record.frames)
                self.rows.append((record, start, stop))
        if max_windows is not None:
            self.rows = self.rows[:max_windows]
        if not self.rows:
            raise ValueError("no target windows")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        record, start, stop = self.rows[index]
        imu, local6d = self.store.slice(record.key, start, stop)
        local = M.r6d_to_rotation_matrix(local6d.reshape(-1, 6)).reshape(-1, 24, 3, 3)
        global_rot = M.forward_kinematics_R(local, self.parent)
        global6d = M.rotation_matrix_to_r6d(global_rot[:, joint_set.reduced].reshape(-1, 3, 3))
        target = global6d.reshape(len(imu), -1)
        return imu, target


def collate(rows):
    lengths = torch.tensor([len(x[0]) for x in rows], dtype=torch.long)
    steps = int(lengths.max())
    imu = rows[0][0].new_zeros((len(rows), steps, 60))
    pose = rows[0][1].new_zeros((len(rows), steps, 96))
    for i, (x, y) in enumerate(rows):
        imu[i, :len(x)] = x
        pose[i, :len(y)] = y
    return imu, pose, lengths


def best_ckpt(root: str) -> str:
    files = list(Path(root).rglob("*.ckpt"))
    if not files:
        raise FileNotFoundError(root)
    return str(sorted(files, key=lambda p: ("validation_step_loss=" not in p.name, p.name))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/pnp5_60a/pnp5")
    ap.add_argument("--audit", default="results/line9/data_audit.json")
    ap.add_argument("--raw-root", default="/home/duanyuhan/dyh/motion/dataset/raw/AMASS")
    ap.add_argument("--imu-store", default="data/line9_store_shape")
    ap.add_argument("--pose-store", default="data/line9_store_pose6d")
    ap.add_argument("--joints-ckpt", default="checkpoints/pnp5_60a_s42_joints")
    ap.add_argument("--poser-ckpt", default="checkpoints/pnp5_60a_s42_poser")
    ap.add_argument("--out", default="checkpoints/line14/latent_adapter.pt")
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    ap.add_argument("--max-windows", type=int, default=0, help="0 means all windows")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.max_batches, a.max_windows = 2, 2, 8
    root = assert_pnp5_data_root(a.data_root)
    validate_pnp5_dataset_metadata(root)
    os.environ.pop("MOBILEPOSER_CAUSAL", None)
    os.environ["MOBILEPOSER_PROCESSED_DATASETS"] = str(root)
    device = torch.device(a.device if a.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(14027)

    records = load_identity_records(root, a.audit, ("train", "validation"), raw_root=a.raw_root,
                                    max_identities_per_fold=3 if a.smoke else None,
                                    max_sequences_per_identity=3 if a.smoke else None)
    store = PoseStore(a.imu_store, a.pose_store)
    max_train = a.max_windows or (8 if a.smoke else None)
    max_val = 4 if a.smoke else None
    train_ds = TargetOnlyWindows(records["train"], store, torch.empty(0), a.window, max_train)
    val_ds = TargetOnlyWindows(records["validation"], store, torch.empty(0), a.window, max_val)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=False, num_workers=0, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, drop_last=False, num_workers=0, collate_fn=collate)
    joints = Joints.load_from_checkpoint(best_ckpt(a.joints_ckpt), map_location=device).to(device).eval()
    poser = Poser.load_from_checkpoint(best_ckpt(a.poser_ckpt), map_location=device).to(device).eval()
    if not hasattr(joints, "joints") or not hasattr(poser, "pose"):
        raise RuntimeError("expected plain PNP5 Joints/Poser RNN checkpoints")
    if joints.joints.linear1.in_features != 60 or joints.joints.linear2.out_features != 72:
        raise RuntimeError("Joints checkpoint is not plain PNP5 60->72")
    if poser.pose.linear1.in_features != 132 or poser.pose.linear2.out_features != 96:
        raise RuntimeError("Poser checkpoint is not plain PNP5 132->96")
    train_ds.parent = joints.bodymodel.parent
    val_ds.parent = joints.bodymodel.parent
    for model in (joints, poser):
        model.requires_grad_(False)
    # cuDNN requires RNN modules in training mode when gradients must flow
    # through them to the adapter input. Keep all baseline weights frozen and
    # disable both dropout layers so this is numerically deterministic.
    for backbone in (joints.joints, poser.pose):
        backbone.train()
        backbone.dropout.p = 0.0
        backbone.rnn.dropout = 0.0
    adapter = CausalLatentSensorAdapter(hidden=128, max_angle=0.65).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=a.lr, weight_decay=1e-4)

    def forward(x, lengths):
        adapted, _, tangent = adapter(x)
        j, _, _ = joints.joints(adapted, lengths)
        p, _, _ = poser.pose(torch.cat((j, adapted), dim=-1), lengths)
        return p, tangent

    def evaluate(loader):
        adapter.eval()
        joints.joints.eval(); poser.pose.eval()
        total = frames = batches = 0
        with torch.no_grad():
            for bi, (imu, target, lengths) in enumerate(loader):
                if a.max_batches and bi >= a.max_batches:
                    break
                imu, target = imu.to(device), target.to(device)
                lengths_dev = lengths.to(device)
                rows = torch.arange(len(lengths), device=device)
                pred, _ = forward(imu, lengths.tolist())
                loss = (pred[rows, lengths_dev - 1] - target[rows, lengths_dev - 1]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite validation loss")
                total += float(loss) * len(lengths); frames += len(lengths); batches += 1
        return total / max(frames, 1), batches

    baseline_val, val_batches = evaluate(val_dl)
    hist = []
    best_val = float("inf")
    for epoch in range(a.epochs):
        adapter.train()
        joints.joints.train(); poser.pose.train()
        total = count = 0
        for bi, (imu, target, lengths) in enumerate(train_dl):
            if a.max_batches and bi >= a.max_batches:
                break
            imu, target = imu.to(device), target.to(device)
            lengths_dev = lengths.to(device)
            end = lengths_dev - 1
            rows = torch.arange(len(lengths), device=device)
            optimizer.zero_grad(set_to_none=True)
            pred, tangent = forward(imu, lengths.tolist())
            loss = (pred[rows, end] - target[rows, end]).square().mean()
            # A small smoothness prior discourages frame-to-frame coordinate jitter.
            valid = torch.arange(imu.shape[1], device=device)[None, :] < lengths_dev[:, None]
            pair = valid[:, 1:] & valid[:, :-1]
            if pair.any():
                loss = loss + 1e-3 * ((tangent[:, 1:] - tangent[:, :-1]).square().mean(-1)[pair].mean())
            if not torch.isfinite(loss):
                details = {"imu_finite": bool(torch.isfinite(imu).all()),
                           "target_finite": bool(torch.isfinite(target).all()),
                           "pred_finite": bool(torch.isfinite(pred).all()),
                           "tangent_finite": bool(torch.isfinite(tangent).all()),
                           "adapter_params_finite": all(bool(torch.isfinite(p).all()) for p in adapter.parameters()),
                           "target_nan": int((~torch.isfinite(target)).sum()),
                           "pred_nan": int((~torch.isfinite(pred)).sum())}
                raise FloatingPointError(f"non-finite adapter loss: {details}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()); count += 1
            print(json.dumps({"epoch": epoch + 1, "batch": bi + 1, "loss": float(loss.detach()),
                              "mean_angle_deg": float(tangent.detach().norm(dim=-1).mean() * 180 / np.pi),
                              "finite": True}), flush=True)
        train_loss = total / max(count, 1)
        val_loss, _ = evaluate(val_dl)
        hist.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                          "baseline_val_loss": baseline_val, "batches": count}), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"adapter": adapter.state_dict(), "hidden": 128, "max_angle": 0.65,
                        "history": hist, "best_epoch": epoch + 1, "best_val_loss": best_val,
                        "baseline_val_loss": baseline_val, "future_frames": 0,
                        "sensors": ["lw", "rw", "ls", "rs", "pelvis"],
                        "joints_ckpt": best_ckpt(a.joints_ckpt), "poser_ckpt": best_ckpt(a.poser_ckpt),
                        "status": "smoke_only" if a.smoke else "trained_not_benchmark_evaluated"}, a.out)

    # Prefix invariance through the complete causal adapter + endpoint-query model.
    adapter.eval()
    x = torch.randn(1, 12, 60, device=device)
    split = 8
    x_changed = x.clone(); x_changed[:, split:] = torch.randn_like(x_changed[:, split:]) * 10
    with torch.no_grad():
        adapted_a, _, _ = adapter(x)
        adapted_b, _, _ = adapter(x_changed)
        future_error = float((adapted_a[:, :split] - adapted_b[:, :split]).abs().max())
        pa, _ = forward(x[:, :split], [split])
        pb, _ = forward(x_changed[:, :split], [split])
        endpoint_error = float((pa[:, -1] - pb[:, -1]).abs().max())
        adapted, _, tangent = adapter(x)
        nan_count = int((~torch.isfinite(adapted)).sum())
    if future_error != 0.0 or endpoint_error != 0.0 or nan_count != 0:
        raise AssertionError({"future_error": future_error, "endpoint_error": endpoint_error, "nan_count": nan_count})
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"saved": str(out), "loss_history": hist, "baseline_val_loss": baseline_val,
                      "best_val_loss": best_val, "val_batches": val_batches,
                      "adapter_future_prefix_max_abs_error": future_error,
                      "pipeline_endpoint_future_max_abs_error": endpoint_error,
                      "nan_count": nan_count, "max_angle_deg": float(tangent.norm(dim=-1).max() * 180 / np.pi)}))


if __name__ == "__main__":
    main()
