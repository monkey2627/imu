"""Rebuild model history from a selected action's stable start-pose hold."""
from __future__ import annotations

import numpy as np
import torch

from production.frame_chain import push_frame, rebuild_feature

DEFAULT_CONTEXT_FRAMES = 60


def corrected_orientation(raw_orientation, calibrator) -> torch.Tensor:
    """Apply the current pose calibration without recording another history frame."""
    orientation = (raw_orientation if isinstance(raw_orientation, torch.Tensor)
                   else torch.from_numpy(np.asarray(raw_orientation, dtype=np.float32)))
    if not calibrator.calibration_enabled:
        return orientation
    headings = torch.from_numpy(calibrator.headings).to(orientation)
    corrections = torch.from_numpy(calibrator.corrections).to(orientation)
    return headings @ orientation @ corrections


def replay_start_context(session, observations, model_id, labels, calibrator,
                         frames: int = DEFAULT_CONTEXT_FRAMES) -> int:
    """Replay the newest stable observations into a fresh inference session.

    Each observation is ``(feature, raw_orientation, aligned_acceleration)``.
    Orientations are recalibrated with the final, atomically committed action
    calibration so the new history never mixes old and new mounting offsets.
    """
    selected = list(observations)[-max(1, int(frames)):]
    if not selected:
        raise ValueError("cannot rebuild model context without observations")
    for feature, raw_orientation, acceleration in selected:
        base = corrected_orientation(raw_orientation, calibrator)
        rebuilt, model_orientation = rebuild_feature(
            feature, model_id, labels, base, calibrator=calibrator)
        accel = (acceleration if isinstance(acceleration, torch.Tensor)
                 else torch.from_numpy(np.asarray(acceleration, dtype=np.float32)))
        push_frame(session, rebuilt, model_orientation, accel)
    if hasattr(session, "finish_replay"):
        session.finish_replay()
    return len(selected)
