"""Train a dense five-IMU pose head over frozen IMUDINO patch features."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from downstream.model import FiveIMUHead
from downstream.prepare import AMASS_NAMES, ATTACHMENT_SHA256, DEFAULT_CACHE_ROOT, REPO_ROOT


class CachedWindowDataset(Dataset):
    """Expose validated feature windows from disjoint source-group folds."""

    def __init__(
        self,
        paths: list[Path],
        fold: str,
        expected_attachment: str = ATTACHMENT_SHA256,
    ) -> None:
        """Load complete feature shards and select one source-group fold.

        Args:
            paths: Dataset feature-cache files made by the frozen encoder.
            fold: Either "train" or "val".
            expected_attachment: Required SMPL surface configuration hash.
        """
        super().__init__()
        # 1. Validate every cache against the physical sensor and temporal contract.
        if fold not in ("train", "val") or not paths:
            raise ValueError("A nonempty cache list and train/val fold are required")
        self.shards: list[dict] = []
        self.indices: list[tuple[int, int]] = []
        self.groups: set[str] = set()
        self.is_birddog: list[bool] = []
        seen_groups: dict[str, bool] = {}
        for path in paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            metadata = shard["metadata"]
            if (
                metadata["sensor_order"] != ["lw", "rw", "ls", "rs", "waist"]
                or metadata["fps"] != 60
                or metadata["attachment_sha256"] != expected_attachment
            ):
                raise ValueError(f"Incompatible feature cache: {path}")
            count = len(shard["features"])
            if not all(len(shard[key]) == count for key in ("targets", "masks", "folds", "groups")):
                raise ValueError(f"Misaligned feature cache: {path}")
            self.shards.append(shard)
            shard_index = len(self.shards) - 1

            # 2. Keep all windows from one motion/video in exactly one fold.
            for row, (group, held_out) in enumerate(zip(shard["groups"], shard["folds"])):
                held_out = bool(held_out)
                if group in seen_groups and seen_groups[group] != held_out:
                    raise ValueError(f"Source group crosses folds: {group}")
                seen_groups[group] = held_out
                if held_out == (fold == "val"):
                    self.indices.append((shard_index, row))
                    self.groups.add(group)
                    self.is_birddog.append(path.stem.startswith("BirdDog"))
        if not self.indices:
            raise ValueError(f"No {fold} windows in feature caches")

    def __len__(self) -> int:
        """Return the number of windows in this fold."""
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return one feature window, frame targets, validity mask, and source category.

        Args:
            index: Position inside the selected fold.
        Returns:
            Features, target rotations, valid-frame flags, and BirdDog indicator.
        """
        # 1. Resolve a window without duplicating the full in-memory shard.
        shard_index, row = self.indices[index]
        shard = self.shards[shard_index]
        return (
            shard["features"][row],
            shard["targets"][row],
            shard["masks"][row],
            self.is_birddog[index],
        )


def masked_rotation_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average 6D rotation error over valid frames and articulated joints only.

    Args:
        prediction: Predicted rotations [B,L,138].
        target: Ground-truth local SMPL rotations [B,L,138].
        mask: Valid source frames [B,L].
        sample_weights: Optional nonnegative weight per motion window [B].
    Returns:
        Scalar mean squared rotation error.
    """
    # 1. Exclude padding while giving each source window equal base importance.
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError("Pose predictions, targets and masks are misaligned")
    valid = mask.float()
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("A pose window cannot contain zero valid frames")
    per_window = ((prediction.float() - target.float()).square() * valid.unsqueeze(-1)).sum(
        dim=(1, 2)
    ) / (counts * prediction.shape[-1])

    # 2. Optionally emphasize BirdDog without dropping any AMASS training window.
    if sample_weights is None:
        return per_window.mean()
    if sample_weights.shape != counts.shape or bool((sample_weights < 0).any()):
        raise ValueError("Expected nonnegative weight per window")
    return (per_window * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def _epoch_train(
    head: FiveIMUHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bird_weight: float,
    epoch: int,
) -> float:
    """Train one full pass over all AMASS and BirdDog windows.

    Args:
        head: Trainable pose regressor.
        loader: Training windows with each source window present once.
        optimizer: AdamW optimizer for head parameters.
        device: Compute device.
        bird_weight: Relative BirdDog pseudo-label loss weight.
        epoch: One-based epoch number for progress logging.
    Returns:
        Mean optimization loss over the epoch.
    """
    # 1. Update only the downstream head using cached frozen features.
    head.train()
    total_loss = 0.0
    total_windows = 0
    for step, (features, targets, masks, is_birddog) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        weights = torch.where(
            is_birddog.to(device),
            torch.full((len(features),), bird_weight, device=device),
            torch.ones(len(features), device=device),
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = head(features, frames=targets.shape[1])
            loss = masked_rotation_mse(prediction, targets, masks, weights)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(features)
        total_windows += len(features)
        if step % 200 == 0:
            print(
                f"[train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={total_loss / total_windows:.6f}",
                flush=True,
            )
    return total_loss / total_windows


def _epoch_validate(
    head: FiveIMUHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Measure held-out pose error separately on AMASS and BirdDog.

    Args:
        head: Dense pose regressor in evaluation mode.
        loader: Windows from source groups absent from training.
        device: Compute device.
    Returns:
        Mean 6D rotation MSE for all data and each available source category.
    """
    # 1. Accumulate valid-frame error without artificial padding or augmentation.
    head.eval()
    totals = {"all": 0.0, "amass": 0.0, "birddog": 0.0}
    counts = {"all": 0, "amass": 0, "birddog": 0}
    with torch.inference_mode():
        for features, targets, masks, is_birddog in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = head(features, frames=targets.shape[1])
            for row, bird in enumerate(is_birddog.tolist()):
                error = float(masked_rotation_mse(
                    prediction[row : row + 1],
                    targets[row : row + 1],
                    masks[row : row + 1],
                ))
                category = "birddog" if bird else "amass"
                totals[category] += error
                counts[category] += 1
                totals["all"] += error
                counts["all"] += 1

    # 2. Report each domain independently so pseudo-label behavior stays visible.
    return {key: totals[key] / counts[key] for key in totals if counts[key]}


def _save_checkpoint(path: Path, state: dict) -> None:
    """Atomically publish a resumable training checkpoint.

    Args:
        path: Latest or best checkpoint path.
        state: Model, optimizer, scheduler, and experiment state.
    """
    # 1. Avoid a partial checkpoint if the process is interrupted while writing.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main() -> None:
    """Train the frozen-backbone five-sensor pose probe from complete feature caches."""
    # 1. Resolve data, device, and a fresh or resumable experiment directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=[*AMASS_NAMES, "BirdDog"])
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs/five_imu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--bird-loss-weight", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.bird_loss_weight <= 0:
        parser.error("Epochs, batch size and BirdDog weight must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    cache_paths = [args.cache_root / f"{name}.features.pt" for name in args.datasets]
    missing = [str(path) for path in cache_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature caches: {missing}")

    # 2. Enforce source-group separation and load each frozen feature window once.
    train = CachedWindowDataset(cache_paths, "train")
    validation = CachedWindowDataset(cache_paths, "val")
    if train.groups & validation.groups:
        raise ValueError("A source motion/video appears in both training and validation")
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(
        f"[data] train={len(train)} val={len(validation)} "
        f"BirdDog_train={sum(train.is_birddog)} BirdDog_val={sum(validation.is_birddog)}",
        flush=True,
    )

    # 3. Optimize only the new probe, resuming safely from the latest epoch if present.
    head = FiveIMUHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "latest.pt"
    best_path = args.output / "best.pt"
    start_epoch = 1
    best_score = math.inf
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["datasets"] != args.datasets or state["seed"] != args.seed:
            raise ValueError("Existing checkpoint belongs to a different dataset or seed")
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_score = state["best_score"]
        print(f"[resume] epoch={start_epoch} best_val={best_score:.6f}", flush=True)

    # 4. Save validation traces and checkpoints after every complete epoch.
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = _epoch_train(
            head, train_loader, optimizer, device, args.bird_loss_weight, epoch
        )
        validation_loss = _epoch_validate(head, val_loader, device)
        score = validation_loss["all"]
        scheduler.step()
        improved = score < best_score
        best_score = min(best_score, score)
        state = {
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "datasets": args.datasets,
            "seed": args.seed,
            "sensor_order": ["lw", "rw", "ls", "rs", "waist"],
            "target": "SMPL local 6D joints 1..23",
            "window_frames": 480,
        }
        _save_checkpoint(latest, state)
        if improved:
            _save_checkpoint(best_path, state)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation_loss,
            "best_validation": best_score,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[epoch] {json.dumps(record, sort_keys=True)}", flush=True)
    (args.output / "COMPLETE").write_text(f"epochs={args.epochs}\n", encoding="utf-8")
    print("[training complete]", flush=True)


if __name__ == "__main__":
    main()
