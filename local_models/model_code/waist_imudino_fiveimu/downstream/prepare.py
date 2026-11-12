"""Build frozen IMUDINO feature caches from shared AMASS-style 5-IMU synthesis."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from downstream.data import group_fold, rotation_targets, to_imu6, window_sequence
from downstream.model import encode_imu
from model_factory import imuformer_huge_patch16


# These are the same eighteen training subsets used by the waist-surface reference.
AMASS_NAMES = (
    "ACCAD", "BioMotionLab_NTroje", "BMLhandball", "BMLmovi", "CMU",
    "DFaust_67", "EKUT", "Eyes_Japan_Dataset", "HumanEva", "KIT",
    "MPI_HDM05", "MPI_Limits", "MPI_mosh", "SFU", "SSM_synced",
    "TCD_handMocap", "TotalCapture", "Transitions_mocap",
)
# The reference surface configuration is tied to one exact SMPL model.
ATTACHMENT_SHA256 = "f18a2bb96442e482fe4faf0f5ca47c157a3d31112e469db9ffda46931f7744f2"
REPO_ROOT = Path(__file__).resolve().parents[1]
AMASS_PROCESSED = Path(
    "/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030/"
    "pnp_workspace/data/amass_waist_surface_60_paper18"
)
BIRDDOG_PROCESSED = REPO_ROOT / "data/processed/wrists_shanks_waist/BirdDog.pt"
BIRDDOG_RAW = Path("/home/duanyuhan/dyh/motion/flowhmr/output/training_data/BirdDog")
DEFAULT_CACHE_ROOT = REPO_ROOT / "data/features/five_imu_v1"


def bird_dog_groups(processed_path: Path, raw_root: Path = BIRDDOG_RAW) -> list[str]:
    """Map each processed BirdDog sequence to its original source-video group.

    Args:
        processed_path: Surface-synthesized BirdDog .pt file.
        raw_root: AMASS-format BirdDog clips with JSON sidecars.
    Returns:
        One stable source-video group key per processed sequence.
    """
    # 1. Recover the sorted clip order used by the shared AMASS reader.
    processed = torch.load(processed_path, map_location="cpu", weights_only=False)
    raw_files = sorted(raw_root.glob("*/*_poses.npz"))
    if len(raw_files) != len(processed["pose"]):
        raise ValueError("BirdDog raw/processed sequence counts differ")

    # 2. Bind every clip to its source video and verify frame alignment.
    groups = []
    for raw_file, pose in zip(raw_files, processed["pose"]):
        sidecar = raw_file.with_name(raw_file.name.replace("_poses.npz", ".json"))
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata["num_frames"] != len(pose):
            raise ValueError(f"BirdDog frame alignment failed: {raw_file}")
        groups.append(f"BirdDog:{metadata['source_video_sha256']}")
    return groups


def _encode_pending(
    pending: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
    backbone: torch.nn.Module,
    device: torch.device,
    feature_chunks: list[torch.Tensor],
    target_chunks: list[torch.Tensor],
    mask_chunks: list[torch.Tensor],
    fold_chunks: list[torch.Tensor],
    window_groups: list[str],
) -> None:
    """Move one bounded batch through the frozen encoder into CPU cache chunks.

    Args:
        pending: IMU, target, mask, and group tuples awaiting extraction.
        backbone: Released encoder, loaded strictly and frozen.
        device: GPU or CPU used for feature extraction.
        feature_chunks: Output patch-feature batches retained on CPU.
        target_chunks: Output pose-target batches retained on CPU.
        mask_chunks: Output valid-frame masks retained on CPU.
        fold_chunks: Output train/validation choices retained on CPU.
        window_groups: Stable source identifiers for audit and leakage checks.
    """
    # 1. Batch windows so the eighty-five-million-parameter encoder is reused.
    if not pending:
        return
    imu = torch.stack([item[0] for item in pending]).to(device)
    masks = torch.stack([item[2] for item in pending]).to(device)

    # 2. Extract patch descriptors with no backbone gradients or optimizer state.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        descriptors = encode_imu(backbone, imu, masks, fps=60)
    feature_chunks.append(descriptors.to(device="cpu", dtype=torch.float16))
    target_chunks.append(torch.stack([item[1] for item in pending]).to(torch.float16))
    mask_chunks.append(torch.stack([item[2] for item in pending]))
    fold_chunks.append(torch.tensor([group_fold(item[3]) == "val" for item in pending], dtype=torch.bool))
    window_groups.extend(item[3] for item in pending)
    pending.clear()


def prepare_one(
    source_path: Path,
    cache_path: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    groups: list[str] | None = None,
    batch_size: int = 8,
    expected_attachment: str = ATTACHMENT_SHA256,
) -> dict:
    """Cache all valid windows from one identically synthesized motion dataset.

    Args:
        source_path: AMASS or BirdDog processed .pt using the common surface pipeline.
        cache_path: New atomic feature-cache artifact.
        backbone: Released IMUDINO model in evaluation mode.
        device: Feature-extraction compute device.
        groups: Optional source-video keys; defaults to sequence identities.
        batch_size: Maximum simultaneous 480-frame windows sent to the encoder.
        expected_attachment: Required SMPL surface-attachment hash.
    Returns:
        Dataset, sequence, frame, window, and split counts.
    """
    # 1. Reject a cache with the wrong physical sensors, sampling rate, or model.
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    synthesis = source["synthesis"]
    if (
        source["layout"] != "wrists_shanks_waist"
        or source["layout_labels"] != ["lw", "rw", "ls", "rs", "waist"]
        or synthesis["mode"] != "surface"
        or synthesis["fps"] != 60
        or synthesis["attachment_sha256"] != expected_attachment
    ):
        raise ValueError(f"Incompatible five-IMU synthesis: {source_path}")
    sequences = source["pose"]
    if len(sequences) != len(source["acc"]) or len(sequences) != len(source["ori"]):
        raise ValueError("Processed pose, acceleration and orientation sequence counts differ")
    if groups is None:
        groups = [f"{source_path.stem}:{index}" for index in range(len(sequences))]
    if len(groups) != len(sequences):
        raise ValueError("Every processed motion needs one split group")

    # 2. Convert all clips by the same gravity, gyro, and rotation-target rules.
    feature_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    fold_chunks: list[torch.Tensor] = []
    window_groups: list[str] = []
    pending: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
    skipped_short = 0
    frames = 0
    for index, (acc, ori, pose, group) in enumerate(
        zip(source["acc"], source["ori"], sequences, groups)
    ):
        if len(pose) != len(acc) or len(pose) != len(ori):
            raise ValueError(f"Sequence {source_path.stem}:{index} has misaligned frames")
        frames += len(pose)
        if len(pose) < 16:
            skipped_short += 1
            continue
        imu = to_imu6(acc, ori, fps=60)
        targets = rotation_targets(pose)
        if not torch.isfinite(targets).all():
            raise ValueError(f"Non-finite pose targets in {source_path.stem}:{index}")
        for _, window_imu, window_target, mask in window_sequence(imu, targets):
            pending.append((window_imu, window_target, mask, group))
            if len(pending) >= batch_size:
                _encode_pending(
                    pending, backbone, device, feature_chunks, target_chunks,
                    mask_chunks, fold_chunks, window_groups,
                )
        if (index + 1) % 250 == 0:
            print(
                f"[prepare] {source_path.stem}: {index + 1}/{len(sequences)} sequences",
                flush=True,
            )
    _encode_pending(
        pending, backbone, device, feature_chunks, target_chunks,
        mask_chunks, fold_chunks, window_groups,
    )
    if not feature_chunks:
        raise ValueError(f"No patch-compatible motions in {source_path}")

    # 3. Write complete features and matching targets only after all clips finish.
    features = torch.cat(feature_chunks)
    targets = torch.cat(target_chunks)
    masks = torch.cat(mask_chunks)
    folds = torch.cat(fold_chunks)
    report = {
        "dataset": source_path.stem,
        "sequences": len(sequences),
        "frames": frames,
        "windows": len(features),
        "train_windows": int((~folds).sum()),
        "val_windows": int(folds.sum()),
        "skipped_short_sequences": skipped_short,
    }
    payload = {
        "features": features,
        "targets": targets,
        "masks": masks,
        "folds": folds,
        "groups": window_groups,
        "metadata": {
            "source": str(source_path),
            "attachment_sha256": expected_attachment,
            "sensor_order": ["lw", "rw", "ls", "rs", "waist"],
            "fps": 60,
            "window_frames": 480,
            "window_stride": 240,
            "target": "SMPL local 6D joints 1..23",
            "report": report,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    print(f"[prepared] {json.dumps(report, sort_keys=True)}", flush=True)
    return report


def main() -> None:
    """Prepare all eighteen reference AMASS sets and BirdDog on one available GPU."""
    # 1. Resolve the released checkpoint and the requested dataset subset.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=[*AMASS_NAMES, "BirdDog"])
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "imudino.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    unknown = set(args.datasets) - set((*AMASS_NAMES, "BirdDog"))
    if unknown:
        parser.error(f"Unknown training datasets: {sorted(unknown)}")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    backbone = imuformer_huge_patch16()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone = backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    # 2. Cache each dataset independently so interruptions do not invalidate earlier work.
    for name in args.datasets:
        source_path = BIRDDOG_PROCESSED if name == "BirdDog" else AMASS_PROCESSED / f"{name}.pt"
        cache_path = args.cache_root / f"{name}.features.pt"
        if cache_path.exists():
            print(f"[skip] existing feature cache: {cache_path}", flush=True)
            continue
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        groups = bird_dog_groups(source_path) if name == "BirdDog" else None
        prepare_one(source_path, cache_path, backbone, device, groups, args.batch_size)
        gc.collect()
    print("[prepare complete] all requested feature caches are present", flush=True)


if __name__ == "__main__":
    main()
