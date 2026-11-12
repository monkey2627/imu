"""Real-time mobileposer inference: turn calibrated per-tracker samples into
the 91-frame (60 fps) sliding window mobile_export.MobilePose expects, and run it.

Reuses the inference-only mobile_export.MobilePose wrapper. It loads the
combined checkpoint (written by ``scripts/convert_sparseimu_checkpoint.py``)
and reconstructs the SMPL kinematic tree from global to local rotations
without importing the historical training model package.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from production.paths import paths
from mobileposer.mobile_export import MobilePose, OUTPUT_INDEX, WINDOW
from production.calibration import HeadingCalibration, bone_orientation
from production.layout import ResolvedLayout
from runtime_contract import (
    ACCELERATION_SCALE,
    FPS as CONTRACT_FPS,
    OUTPUT_INDEX as CONTRACT_OUTPUT_INDEX,
    OUTPUT_LAG_FRAMES as CONTRACT_OUTPUT_LAG_FRAMES,
    REALTIME_LAYOUT,
    WINDOW as CONTRACT_WINDOW,
)

# Combined checkpoints live under checkpoints/<model name>/{base_model.pth,manifest.json}.
DEFAULT_CHECKPOINT_ROOT = paths.checkpoint
DEFAULT_MODEL = "pnp5_60a_s42_ftr3"

# The model's pose output at window index OUTPUT_INDEX (80 of 91) reconstructs
# the pose (WINDOW - 1 - OUTPUT_INDEX) = 10 frames behind the most recent
# input frame -- about 167 ms of inherent lag at 60 Hz. This is a property of
# the trained model (online "win46": 40 past + 5 future frames at 30 fps), not
# a bug to fix here.
OUTPUT_LAG_FRAMES = WINDOW - 1 - OUTPUT_INDEX
if (WINDOW, OUTPUT_INDEX) != (CONTRACT_WINDOW, CONTRACT_OUTPUT_INDEX):
    raise RuntimeError("mobile_export and realtime contract window constants diverged")
if OUTPUT_LAG_FRAMES != CONTRACT_OUTPUT_LAG_FRAMES:
    raise RuntimeError("mobile_export and realtime contract lag constants diverged")


class CheckpointError(RuntimeError):
    """The checkpoint does not match the realtime contract."""


def resolve_checkpoint(spec, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT) -> Path:
    """``spec``: a base_model.pth path, a directory containing it, or a bare model name."""
    path = Path(spec)
    if path.suffix == ".pth":
        return path
    if path.is_dir():
        return path / "base_model.pth"
    return Path(checkpoint_root) / str(spec) / "base_model.pth"


def read_manifest(checkpoint: Path) -> dict:
    manifest = Path(checkpoint).parent / "manifest.json"
    if not manifest.is_file():
        raise CheckpointError(
            f"{manifest} is missing. Checkpoints must be produced by "
            "scripts/convert_sparseimu_checkpoint.py (old 30 fps / old-placement "
            "checkpoints are not compatible with the current realtime path)."
        )
    return json.loads(manifest.read_text(encoding="utf-8"))


def check_manifest(manifest: dict, checkpoint: Path) -> None:
    expected = {"layout": REALTIME_LAYOUT, "fps": CONTRACT_FPS, "window": WINDOW, "outputIndex": OUTPUT_INDEX}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointError(
                f"{checkpoint}: manifest {key}={manifest.get(key)!r}, runtime expects {value!r}"
            )


def load_model(spec=DEFAULT_MODEL, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT, anchor: bool | None = None) -> MobilePose:
    """Load a converted checkpoint. ``anchor`` None -> follow the manifest; an
    explicit value that disagrees with the manifest is an error (an anchored
    network and a plain one produce differently-meaning outputs)."""
    path = resolve_checkpoint(spec, checkpoint_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run scripts/convert_sparseimu_checkpoint.py "
            f"on the training server and copy checkpoints/<name>/ here."
        )
    manifest = read_manifest(path)
    check_manifest(manifest, path)
    anchored = bool(manifest.get("anchored", False))
    if anchor is not None and bool(anchor) != anchored:
        raise CheckpointError(f"{path}: manifest anchored={anchored} but anchor={anchor} was requested")
    return MobilePose(anchor=anchored).load_combined(path)


def build_feature(calibrated_rotation: torch.Tensor, aligned_accel: torch.Tensor) -> np.ndarray:
    """calibrated_rotation: (n_labels, 3, 3); aligned_accel: (n_labels, 3) -> (60,) float32.

    Matches the training layout (data.py::_process_combo_data): all five
    accelerations concatenated first (scaled by 1/30, amass.acc_scale), THEN
    all five orientations -- NOT interleaved per sensor slot, even though the
    model is often described as "5 x 12D".
    """
    acc = (aligned_accel.reshape(-1) / ACCELERATION_SCALE).numpy()
    ori = calibrated_rotation.reshape(-1).numpy()
    return np.concatenate([acc, ori]).astype(np.float32)


class SlidingWindow:
    """Push-based re-implementation of the training/eval window logic (that
    helper is an iterator over a whole pre-collected sequence, which doesn't
    fit a live push-one-frame-at-a-time loop)."""

    def __init__(self, size: int = WINDOW) -> None:
        self.size = size
        self._window: np.ndarray | None = None

    def push(self, frame: np.ndarray) -> np.ndarray:
        """frame: (60,) float32 -> (1, size, 60) float32, ready for the model."""
        if self._window is None:
            self._window = np.repeat(frame[None], self.size, axis=0)
        else:
            self._window = np.concatenate([self._window[1:], frame[None]])
        return self._window[None].astype(np.float32)


@dataclasses.dataclass
class InferenceSession:
    """One loaded model + its sliding window, for one resolved layout.

    Orientation conversion is stateless per-frame (see calibration.py). The
    live path uses SlimeVR's reference-adjusted acceleration directly, so the
    heading is the identity (see run.py).
    """

    layout: ResolvedLayout
    model: MobilePose
    heading: HeadingCalibration
    window: SlidingWindow = dataclasses.field(default_factory=SlidingWindow)

    def prime(self, quats_xyzw: np.ndarray, accel_xyz: np.ndarray) -> None:
        """Initialize the recurrent context from a known upright sample."""
        ori = bone_orientation(quats_xyzw, self.layout.labels)
        accel = self.heading.apply(accel_xyz)
        feature = build_feature(ori, accel)
        self.window._window = np.repeat(feature[None], self.window.size, axis=0)

    def prepare_feature(self, quats_xyzw: np.ndarray, accel_xyz: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Return the exact 60-D model feature and calibrated components."""
        ori = bone_orientation(quats_xyzw, self.layout.labels)
        accel = self.heading.apply(accel_xyz)
        return build_feature(ori, accel), ori, accel

    def push_feature(self, feature: np.ndarray) -> None:
        """Advance the window by one 60 Hz frame without running the network."""
        self.window.push(np.asarray(feature, dtype=np.float32))

    def infer(self) -> torch.Tensor:
        """Run the network on the current window -> (24, 3, 3) local rotations."""
        with torch.no_grad():
            return self.model(torch.from_numpy(self.window._window[None].astype(np.float32)))

    def step_feature(self, feature: np.ndarray) -> torch.Tensor:
        """Push a precomputed feature and run the model."""
        window = self.window.push(np.asarray(feature, dtype=np.float32))
        with torch.no_grad():
            return self.model(torch.from_numpy(window))

    def step(self, quats_xyzw: np.ndarray, accel_xyz: np.ndarray) -> torch.Tensor:
        """quats_xyzw: (n_labels, 4) rotation_reference_adjusted quats;
        accel_xyz: (n_labels, 3) linear acceleration (m/s^2, gravity removed)
        straight from SolarXR -- both in self.layout.labels order. Returns
        (24, 3, 3) local rotation matrices for the current window; see
        OUTPUT_LAG_FRAMES for the inherent delay.
        """
        feature, _, _ = self.prepare_feature(quats_xyzw, accel_xyz)
        return self.step_feature(feature)
