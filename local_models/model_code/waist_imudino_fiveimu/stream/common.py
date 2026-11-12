"""Shared streaming contract for the stateful five-IMU IMUDINO head.

Training (feature caches, dataset) and deployment (online adapter) both import
this module, so the token schedule and per-frame features cannot drift apart.

Schedule, in frames of one continuous 60 Hz stream starting at frame 0:
* a backbone token is computed for every grid frame g with g % 16 == 15, from the
  480-frame window ending at g (left part masked when the stream is shorter);
* gyro of frame g needs frame g+1, and one more frame is reserved for compute, so
  token g is usable from frame g + TOKEN_DELAY (= g+2) onward;
* every frame t is decoded from the newest usable token plus frame t's own sensors.
"""
import torch

PATCH = 16
WINDOW = 480
TOKEN_DELAY = 2
SENSORS = 5
TOKEN_DIM = SENSORS * 768
FRAME_DIM = SENSORS * 6 + 4 * 9 + 3 + 2  # imu6, limb-in-waist rotations, waist gravity, phase+valid
GRAVITY_DIR = (0.0, -1.0, 0.0)  # Y-up world, same as downstream.data.GRAVITY_WORLD


def grid_frames(length: int) -> torch.Tensor:
    """Grid frames g (window ends) whose token is usable inside a stream of `length` frames."""
    return torch.arange(PATCH - 1, max(length - TOKEN_DELAY, 0), PATCH)


def token_slot(frames: torch.Tensor) -> torch.Tensor:
    """Index into grid_frames() of the newest usable token per frame, -1 if none yet."""
    return torch.div(frames - TOKEN_DELAY - (PATCH - 1), PATCH, rounding_mode="floor").clamp_min(-1)


def window_bounds(g: int) -> tuple[int, int]:
    """Inclusive-exclusive frame range [start, g+1) of the window ending at grid frame g."""
    return g + 1 - WINDOW, g + 1


def frame_features(imu6: torch.Tensor, ori: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Per-frame fast-path input, yaw invariant.

    Args:
        imu6: [T,5,6] body-frame specific force (m/s^2) + gyro (rad/s) from downstream.data.to_imu6.
        ori: [T,5,3,3] calibrated sensor-to-world rotations in order lw,rw,ls,rs,waist.
        frames: [T] absolute stream frame numbers (for token phase).
    Returns:
        [T, FRAME_DIM] float32.
    """
    waist = ori[:, 4]
    rel = waist.transpose(-1, -2).unsqueeze(1) @ ori[:, :4]  # limb orientation in waist frame
    gravity = (waist.transpose(-1, -2) @ ori.new_tensor(GRAVITY_DIR)).reshape(-1, 3)
    slot = token_slot(frames)
    valid = (slot >= 0).float()
    newest_g = slot.clamp_min(0) * PATCH + PATCH - 1
    phase = ((frames - newest_g - TOKEN_DELAY).float() / PATCH).clamp(0, 1) * valid
    scaled = torch.cat((imu6[..., :3] / 9.81, imu6[..., 3:] / 4.0), dim=-1)
    return torch.cat((scaled.reshape(len(imu6), -1), rel.reshape(len(imu6), -1), gravity,
                      phase[:, None], valid[:, None]), dim=-1).float()
