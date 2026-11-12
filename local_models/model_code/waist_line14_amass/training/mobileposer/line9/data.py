"""Identity-isolated AMASS data for the Line 9 Oracle experiment.

The public :class:`mobileposer.models.joints.Joints` target is deliberately
reconstructed here: it runs FK from the cached local rotations with both
``shape=None`` and ``tran=None``.  In particular, the cached ``joint`` field is
*not* a valid substitute: it contains the motion translation and was generated
with the sequence's SMPL shape.

Nothing in this module depends on a model variant or a training seed.  A
``PairedWindowDataset`` therefore exposes exactly the same target windows and
calibration/target pairs to B0, B1, and Oracle-geometry.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


DEFAULT_RAW_ROOT = Path("data/raw/AMASS")
DEFAULT_PROCESSED_ROOT = Path("data/pnp5_60a/pnp5")
DEFAULT_AUDIT = Path("results/line9/data_audit.json")


def _beta_key(beta: np.ndarray | torch.Tensor, decimals: int = 4) -> str:
    if isinstance(beta, torch.Tensor):
        beta = beta.detach().cpu().numpy()
    value = np.round(np.asarray(beta[:10], dtype=np.float64), decimals)
    return hashlib.sha1(value.tobytes()).hexdigest()[:12]


@dataclass(frozen=True)
class SequenceRecord:
    """One audited raw motion and its exactly corresponding cache row."""

    fold: str
    subset: str
    identity: str
    sequence: str
    raw_path: Path
    cache_path: Path
    row: int
    frames: int
    beta: tuple[float, ...]

    @property
    def key(self) -> str:
        return f"{self.subset}:{self.row}:{self.sequence}"


def _audited_raw_rows(raw_root: Path, subset: str, target_fps: int) -> List[tuple[Path, np.ndarray, int]]:
    """Reproduce the accepted-row ordering used by ``line9_data_audit.py``."""
    rows: List[tuple[Path, np.ndarray]] = []
    for path in sorted((raw_root / subset).glob("*/*_poses.npz")):
        try:
            with np.load(path) as blob:
                fps = int(blob["mocap_framerate"])
                frames = int(len(blob["poses"]))
                beta = np.asarray(blob["betas"][:10], dtype=np.float32).copy()
        except Exception:
            continue
        if fps < target_fps:
            continue
        if int(math.floor(frames / float(fps) * target_fps)) <= 12:
            continue
        step = max(1, round(fps / target_fps))
        if fps in (120, 60, 59):
            cached_frames = (frames + step - 1) // step
        else:
            cached_frames = int(math.floor((frames - 1) * target_fps / float(fps) + 1e-9)) + 1
        rows.append((path, beta, cached_frames))
    return rows


def load_identity_records(
    processed_root: Path | str = DEFAULT_PROCESSED_ROOT,
    audit_path: Path | str = DEFAULT_AUDIT,
    folds: Sequence[str] = ("train", "validation"),
    *,
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    verify: bool = True,
    max_identities_per_fold: Optional[int] = None,
    max_sequences_per_identity: Optional[int] = None,
) -> Dict[str, List[SequenceRecord]]:
    """Rebuild the immutable identity split and raw-to-cache row mapping.

    Limits are solely for engineering smoke tests and are applied after the
    fixed audit split.  With no limits, this function asserts the audit's exact
    identity and sequence counts (342/11,198 train and 49/1,558 validation in
    the current manifest).
    """
    processed_root, audit_path, raw_root = map(Path, (processed_root, audit_path, raw_root))
    report = json.loads(audit_path.read_text())
    split = report["proposed_identity_split"]
    unknown = set(folds) - set(split)
    if unknown:
        raise ValueError(f"unknown folds in audit manifest: {sorted(unknown)}")

    fold_ids: Dict[str, List[str]] = {fold: list(split[fold]["items"]) for fold in folds}
    if max_identities_per_fold is not None:
        if max_identities_per_fold < 1:
            raise ValueError("max_identities_per_fold must be positive")
        fold_ids = {fold: ids[:max_identities_per_fold] for fold, ids in fold_ids.items()}
    identity_fold = {identity: fold for fold, ids in fold_ids.items() for identity in ids}
    if len(identity_fold) != sum(map(len, fold_ids.values())):
        raise AssertionError("the audit manifest leaks at least one identity across folds")

    target_fps = int(report["protocol"]["target_fps"])
    excluded = set(report["protocol"].get("excluded_subsets", ())) | {"TotalCapture"}
    wanted_subsets = sorted({identity.split("/", 1)[0] for identity in identity_fold})
    records: Dict[str, List[SequenceRecord]] = {fold: [] for fold in folds}

    for subset in wanted_subsets:
        if subset in excluded:
            raise AssertionError(f"excluded subset unexpectedly appears in split: {subset}")
        cache_path = processed_root / f"{subset}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        raw_rows = _audited_raw_rows(raw_root, subset, target_fps)
        expected_rows = int(report["source_count_check"][subset]["processed"])
        if len(raw_rows) != expected_rows:
            raise AssertionError(
                f"raw/cache row mismatch for {subset}: {len(raw_rows)} != {expected_rows}"
            )
        for row, (raw_path, raw_beta, frames) in enumerate(raw_rows):
            identity = f"{subset}/{raw_path.parent.name}"
            fold = identity_fold.get(identity)
            if fold is None:
                continue
            records[fold].append(
                SequenceRecord(
                    fold=fold,
                    subset=subset,
                    identity=identity,
                    sequence=str(raw_path.relative_to(raw_root)),
                    raw_path=raw_path,
                    cache_path=cache_path,
                    row=row,
                    frames=frames,
                    beta=tuple(float(x) for x in raw_beta.tolist()),
                )
            )

    for fold in folds:
        records[fold].sort(key=lambda r: (r.identity, r.sequence, r.row))
        grouped: Dict[str, List[SequenceRecord]] = defaultdict(list)
        for record in records[fold]:
            grouped[record.identity].append(record)
        missing = set(fold_ids[fold]) - set(grouped)
        if missing:
            raise AssertionError(f"{fold} identities absent from cache mapping: {sorted(missing)[:5]}")
        if min(map(len, grouped.values())) < 2:
            raise AssertionError(f"{fold} contains an identity with fewer than two motions")
        if max_sequences_per_identity is not None:
            if max_sequences_per_identity < 2:
                raise ValueError("max_sequences_per_identity must be at least two for cross-motion pairing")
            records[fold] = [r for identity in sorted(grouped) for r in grouped[identity][:max_sequences_per_identity]]

        unrestricted = max_identities_per_fold is None and max_sequences_per_identity is None
        if unrestricted:
            expected_ids = int(split[fold]["identities"])
            expected_sequences = int(split[fold]["sequences"])
            actual_ids = len({r.identity for r in records[fold]})
            if (actual_ids, len(records[fold])) != (expected_ids, expected_sequences):
                raise AssertionError(
                    f"{fold} manifest mismatch: got {actual_ids}/{len(records[fold])}, "
                    f"expected {expected_ids}/{expected_sequences} identities/sequences"
                )

    assert_identity_isolation(records)
    return records


def assert_identity_isolation(records_by_fold: Mapping[str, Sequence[SequenceRecord]]) -> None:
    """Fail loudly on identity, raw-sequence, or cache-row leakage."""
    identities: Dict[str, set[str]] = {}
    sequences: Dict[str, set[str]] = {}
    cache_rows: Dict[str, set[tuple[str, int]]] = {}
    for fold, rows in records_by_fold.items():
        identities[fold] = {r.identity for r in rows}
        sequences[fold] = {r.sequence for r in rows}
        cache_rows[fold] = {(str(r.cache_path.resolve()), r.row) for r in rows}
        if len(sequences[fold]) != len(rows) or len(cache_rows[fold]) != len(rows):
            raise AssertionError(f"duplicate sequence/cache row inside {fold}")
        if any(r.fold != fold for r in rows):
            raise AssertionError(f"record with incorrect fold label inside {fold}")
    names = list(records_by_fold)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if overlap := identities[left] & identities[right]:
                raise AssertionError(f"identity leakage {left}/{right}: {sorted(overlap)[:5]}")
            if sequences[left] & sequences[right] or cache_rows[left] & cache_rows[right]:
                raise AssertionError(f"sequence leakage between {left} and {right}")


@dataclass(frozen=True)
class _Window:
    calibration: SequenceRecord
    target: SequenceRecord
    start: int
    stop: int


class PairedWindowDataset(Dataset):
    """Deterministic cross-motion episodes with public-Joints-equivalent targets.

    Every sequence remains a target.  The lexicographically first motion is the
    identity's fixed calibration motion, except when that motion itself is the
    target, in which case the second motion is used.  This preserves all audited
    target data while guaranteeing different calibration and target sequences.
    """

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        window_size: int = 125,
        *,
        calibration_window_size: Optional[int] = None,
        include_tail: bool = True,
        max_windows: Optional[int] = None,
        file_cache_size: int = 1,
        sequence_cache_size: int = 32,
        store=None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if file_cache_size < 1 or sequence_cache_size < 1:
            raise ValueError("cache sizes must be positive")
        self.window_size = int(window_size)
        self.calibration_window_size = int(calibration_window_size or window_size)
        self.file_cache_size = int(file_cache_size)
        self.sequence_cache_size = int(sequence_cache_size)
        self._files: OrderedDict[str, dict] = OrderedDict()
        self._sequences: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._poses: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._geometry: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._body_model = None
        self.store = store

        grouped: Dict[str, List[SequenceRecord]] = defaultdict(list)
        for record in records:
            grouped[record.identity].append(record)
        self.windows: List[_Window] = []
        for identity in sorted(grouped):
            motions = sorted(grouped[identity], key=lambda r: (r.sequence, r.row))
            if len(motions) < 2:
                raise AssertionError(f"identity lacks a distinct calibration motion: {identity}")
            for target in motions:
                calibration = motions[0] if target != motions[0] else motions[1]
                if calibration.identity != target.identity or calibration.key == target.key:
                    raise AssertionError("invalid cross-motion calibration pair")
                starts = range(0, target.frames, self.window_size)
                for start in starts:
                    stop = min(start + self.window_size, target.frames)
                    if include_tail or stop - start == self.window_size:
                        self.windows.append(_Window(calibration, target, start, stop))
        if max_windows is not None:
            if max_windows < 1:
                raise ValueError("max_windows must be positive")
            self.windows = self.windows[:max_windows]
        if not self.windows:
            raise ValueError("no target windows were constructed")
        self._assert_pairs()

    def _assert_pairs(self) -> None:
        for item in self.windows:
            if item.calibration.identity != item.target.identity:
                raise AssertionError("calibration/target identity mismatch")
            if item.calibration.key == item.target.key:
                raise AssertionError("calibration and target use the same sequence")
            if not (0 <= item.start < item.stop <= item.target.frames):
                raise AssertionError("invalid target window bounds")

    @property
    def target_window_keys(self) -> tuple[tuple[str, int, int], ...]:
        """Stable fingerprint input for asserting fairness across variants/seeds."""
        return tuple((w.target.key, w.start, w.stop) for w in self.windows)

    def target_window_digest(self) -> str:
        payload = "\n".join(f"{key}:{start}:{stop}" for key, start, stop in self.target_window_keys)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_file(self, path: Path) -> dict:
        key = str(path.resolve())
        if key not in self._files:
            self._files[key] = torch.load(path, map_location="cpu", weights_only=False)
            while len(self._files) > self.file_cache_size:
                self._files.popitem(last=False)
        else:
            self._files.move_to_end(key)
        return self._files[key]

    def _body(self):
        if self._body_model is None:
            from mobileposer.articulate.model import ParametricModel
            from mobileposer.config import paths

            self._body_model = ParametricModel(paths.smpl_file, device=torch.device("cpu"))
        return self._body_model

    @torch.no_grad()
    def _get_sequence(self, record: SequenceRecord) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized five-IMU input and mean-shape/no-translation FK target."""
        if self.store is not None:
            return self.store.get(record.key)
        if record.key not in self._sequences:
            cache = self._get_file(record.cache_path)
            cached_beta = cache["shape"][record.row].float().reshape(-1)[:10]
            if _beta_key(cached_beta) != _beta_key(torch.tensor(record.beta)):
                raise AssertionError(f"beta mismatch at {record.subset} row {record.row}")
            acc = cache["acc"][record.row].float()[:, :5] / 30.0
            ori = cache["ori"][record.row].float()[:, :5]
            imu = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1).contiguous()
            pose = cache["pose"][record.row].float()
            if not (len(pose) == len(acc) == len(ori) == record.frames):
                raise AssertionError(f"unaligned cache fields at {record.subset} row {record.row}")
            # This is exactly PoseDataset._process_file_data's Joints target.
            _, joints = self._body().forward_kinematics(
                pose=pose.reshape(-1, 216), shape=None, tran=None, calc_mesh=False
            )
            value = (imu, joints.reshape(-1, 24, 3).contiguous())
            self._sequences[record.key] = value
            while len(self._sequences) > self.sequence_cache_size:
                self._sequences.popitem(last=False)
        else:
            self._sequences.move_to_end(record.key)
        return self._sequences[record.key]

    @torch.no_grad()
    def _get_global_pose(self, record: SequenceRecord) -> torch.Tensor:
        """Return the same global rotation targets used by MobilePoser Poser."""
        if record.key not in self._poses:
            cache = self._get_file(record.cache_path)
            local = cache["pose"][record.row].float().reshape(-1, 216)
            global_pose, _ = self._body().forward_kinematics(
                pose=local, shape=None, tran=None, calc_mesh=False
            )
            self._poses[record.key] = global_pose.reshape(-1, 24, 3, 3).contiguous()
            while len(self._poses) > self.sequence_cache_size:
                self._poses.popitem(last=False)
        else:
            self._poses.move_to_end(record.key)
        return self._poses[record.key]

    @torch.no_grad()
    def _get_geometry(self, record: SequenceRecord) -> tuple[torch.Tensor, torch.Tensor]:
        if record.identity not in self._geometry:
            beta = torch.tensor(record.beta, dtype=torch.float32)
            joints, _ = self._body().get_zero_pose_joint_and_vertex(beta)
            joints = joints.reshape(24, 3)
            parents = self._body().parent
            lengths = torch.zeros(24, dtype=torch.float32)
            for joint in range(1, 24):
                lengths[joint] = (joints[joint] - joints[parents[joint]]).norm()
            self._geometry[record.identity] = (beta, lengths)
        return self._geometry[record.identity]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        item = self.windows[index]
        beta, bone_lengths = self._get_geometry(item.target)
        if self.store is not None:
            calibration_imu, _ = self.store.slice(item.calibration.key, 0, self.calibration_window_size)
            target_imu, target_joints = self.store.slice(item.target.key, item.start, item.stop)
            offset = item.start
        else:
            calibration_imu, _ = self._get_sequence(item.calibration)
            target_imu, target_joints = self._get_sequence(item.target)
            calibration_imu = calibration_imu[: self.calibration_window_size]
            offset = 0
        return {
            "calibration_imu": calibration_imu,
            "target_imu": target_imu[item.start - offset : item.stop - offset],
            "target_joints": target_joints[item.start - offset : item.stop - offset],
            "target_pose": self._get_global_pose(item.target)[item.start : item.stop],
            "beta": beta,
            "bone_lengths": bone_lengths,
            "identity": item.target.identity,
            "calibration_sequence": item.calibration.sequence,
            "target_sequence": item.target.sequence,
            "target_subset": item.target.subset,
            "window_start": item.start,
            "window_stop": item.stop,
        }


def _pad_first_dim(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    steps = int(lengths.max())
    output = values[0].new_zeros((len(values), steps, *values[0].shape[1:]))
    for index, value in enumerate(values):
        output[index, : len(value)] = value
    mask = torch.arange(steps).unsqueeze(0) < lengths.unsqueeze(1)
    return output, lengths, mask


def collate_paired_windows(batch: Sequence[dict]) -> dict:
    """Pad calibration and tail target windows without losing provenance."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    calibration, calibration_lengths, calibration_mask = _pad_first_dim(
        [item["calibration_imu"] for item in batch]
    )
    target, target_lengths, target_mask = _pad_first_dim([item["target_imu"] for item in batch])
    joints, joint_lengths, joint_mask = _pad_first_dim([item["target_joints"] for item in batch])
    poses, pose_lengths, pose_mask = _pad_first_dim([item["target_pose"] for item in batch])
    if not torch.equal(target_lengths, joint_lengths) or not torch.equal(target_mask, joint_mask) or not torch.equal(target_mask, pose_mask):
        raise AssertionError("target IMU and FK supervision lengths diverged")
    for item in batch:
        if item["calibration_sequence"] == item["target_sequence"]:
            raise AssertionError("same-sequence calibration leakage in collate")
    return {
        "calibration_imu": calibration,
        "calibration_lengths": calibration_lengths,
        "calibration_mask": calibration_mask,
        "target_imu": target,
        "target_joints": joints,
        "target_pose": poses,
        "target_lengths": target_lengths,
        "target_mask": target_mask,
        "beta": torch.stack([item["beta"] for item in batch]),
        "bone_lengths": torch.stack([item["bone_lengths"] for item in batch]),
        "identity": tuple(item["identity"] for item in batch),
        "calibration_sequence": tuple(item["calibration_sequence"] for item in batch),
        "target_sequence": tuple(item["target_sequence"] for item in batch),
        "target_subset": tuple(item["target_subset"] for item in batch),
        "window_start": torch.tensor([item["window_start"] for item in batch], dtype=torch.long),
        "window_stop": torch.tensor([item["window_stop"] for item in batch], dtype=torch.long),
    }


class SubsetBatchSampler(Sampler[List[int]]):
    """Shuffle windows while keeping cache-file accesses locally grouped.

    Loading a legacy ``.pt`` subset necessarily deserializes the whole file.
    Grouping batches by subset avoids repeatedly loading multi-GB subsets when
    ``PairedWindowDataset.file_cache_size`` is small.  ``set_epoch`` provides
    the same deterministic behavior as distributed samplers.
    """

    def __init__(
        self,
        dataset: PairedWindowDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        grouped: Dict[str, List[int]] = defaultdict(list)
        for index, window in enumerate(dataset.windows):
            grouped[window.target.subset].append(index)
        self.groups = dict(sorted(grouped.items()))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        subsets = list(self.groups)
        if self.shuffle:
            rng.shuffle(subsets)
        for subset in subsets:
            indices = list(self.groups[subset])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            total += len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size:
                total += 1
        return total


def assert_same_target_windows(*datasets: PairedWindowDataset) -> None:
    """Assert that model variants/seeds receive an identical target corpus."""
    if len(datasets) < 2:
        return
    reference = datasets[0].target_window_keys
    for index, dataset in enumerate(datasets[1:], start=1):
        if dataset.target_window_keys != reference:
            raise AssertionError(f"dataset {index} has different target windows/pair ordering")


__all__ = [
    "SequenceRecord",
    "load_identity_records",
    "assert_identity_isolation",
    "PairedWindowDataset",
    "collate_paired_windows",
    "SubsetBatchSampler",
    "assert_same_target_windows",
]
