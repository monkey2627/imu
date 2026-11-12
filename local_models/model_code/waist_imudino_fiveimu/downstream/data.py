"""Convert common SMPL-synthesized kinematics into IMUDINO training samples."""

import hashlib
from collections.abc import Iterator

import torch
import torch.nn.functional as F


# The sixth slot in the MobilePoser cache is a compatibility root, not a mounted IMU.
SENSOR_COUNT = 5
# The MobilePoser synthesis rotates AMASS from Z-up into this Y-up world frame.
GRAVITY_WORLD = (0.0, -9.81, 0.0)


def to_imu6(acc_world: torch.Tensor, ori_world: torch.Tensor, fps: int) -> torch.Tensor:
    """Return five body-frame accelerometer/gyroscope streams in SI units.

    Args:
        acc_world: Linear world acceleration, shape [T, at least 5, 3], m/s?.
        ori_world: Calibrated sensor-to-world rotation, shape [T, at least 5, 3, 3].
        fps: Positive sample frequency in Hz.
    Returns:
        Six-axis streams [T, 5, 6] with acceleration in m/s? and gyro in rad/s.
    """
    # 1. Reject mismatched clips before their channels can silently be reordered.
    if fps <= 0 or acc_world.ndim != 3 or ori_world.ndim != 4:
        raise ValueError("Expected positive fps and [T,S,3]/[T,S,3,3] tensors")
    if acc_world.shape[0] != ori_world.shape[0] or acc_world.shape[1] < SENSOR_COUNT:
        raise ValueError("Acceleration and orientation need the same frames and five sensors")
    if acc_world.shape[-1] != 3 or ori_world.shape[-2:] != (3, 3) or len(acc_world) == 0:
        raise ValueError("Invalid acceleration or orientation dimensions")

    # 2. Rotate gravity-retained specific force into each calibrated sensor frame.
    rotations = ori_world[:, :SENSOR_COUNT].float()
    linear_acceleration = acc_world[:, :SENSOR_COUNT].float()
    gravity = linear_acceleration.new_tensor(GRAVITY_WORLD)
    specific_force = linear_acceleration - gravity
    acc_body = (rotations.transpose(-1, -2) @ specific_force.unsqueeze(-1)).squeeze(-1)

    # 3. Derive angular velocity from full-sequence relative rotations before cropping.
    gyro = torch.zeros_like(acc_body)
    if len(rotations) > 1:
        relative = rotations[:-1].transpose(-1, -2) @ rotations[1:]
        vee = 0.5 * torch.stack(
            (
                relative[..., 2, 1] - relative[..., 1, 2],
                relative[..., 0, 2] - relative[..., 2, 0],
                relative[..., 1, 0] - relative[..., 0, 1],
            ),
            dim=-1,
        )
        sine = torch.linalg.vector_norm(vee, dim=-1)
        cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
        angle = torch.atan2(sine, cosine)
        # WHY: The series limit avoids 0/0 for a still sensor while retaining small motions.
        scale = torch.where(
            sine > 1e-5,
            angle / sine.clamp_min(1e-8),
            1.0 + angle.square() / 6.0,
        )
        gyro[:-1] = vee * scale.unsqueeze(-1) * fps
        gyro[-1] = gyro[-2]

    # 4. Keep the channel contract shared by AMASS and BirdDog.
    imu = torch.cat((acc_body, gyro), dim=-1)
    if not torch.isfinite(imu).all():
        raise ValueError("Non-finite six-axis IMU sample")
    return imu


def rotation_targets(pose_local: torch.Tensor) -> torch.Tensor:
    """Return local 6D SMPL rotations for joints 1..23, excluding world root yaw.

    Args:
        pose_local: Twenty-four local joint rotation matrices [T,24,3,3].
    Returns:
        Per-frame rotation targets [T,138].
    """
    # 1. Keep the articulated joints while excluding the unobservable world root.
    if pose_local.ndim != 4 or pose_local.shape[1:] != (24, 3, 3):
        raise ValueError("Expected local SMPL rotations [T,24,3,3]")
    child_rotations = pose_local[:, 1:].float()

    # 2. Use the first two rotation columns as a continuous 6D target.
    return child_rotations[..., :2].transpose(-1, -2).reshape(len(pose_local), 23 * 6)


def window_sequence(
    imu: torch.Tensor,
    target: torch.Tensor,
    length: int = 480,
    stride: int = 240,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield padded fixed-size windows while covering the final frame of every clip.

    Args:
        imu: Five six-axis sensor streams [T,5,6].
        target: Per-frame rotation targets [T,138].
        length: Window frames, positive and divisible by the encoder patch size.
        stride: Positive step between complete windows.
    Yields:
        Start frame, padded IMU, padded target, and valid-frame mask.
    """
    # 1. Validate temporal alignment and patch-compatible window size.
    if length < 16 or length % 16 or stride < 1 or len(imu) != len(target):
        raise ValueError("Window must be patch-aligned and IMU/target lengths must agree")
    if len(imu) == 0:
        return

    # 2. Include a terminal window so no valid pose frame is discarded.
    starts = list(range(0, max(len(imu) - length + 1, 1), stride))
    final_start = max(0, len(imu) - length)
    if starts[-1] != final_start:
        starts.append(final_start)
    for start in starts:
        valid = min(length, len(imu) - start)
        imu_window = F.pad(imu[start : start + valid], (0, 0, 0, 0, 0, length - valid))
        target_window = F.pad(target[start : start + valid], (0, 0, 0, length - valid))
        mask = torch.arange(length) < valid
        yield start, imu_window, target_window, mask


def group_fold(group: str, seed: int = 42) -> str:
    """Assign one motion or source video to train or validation deterministically.

    Args:
        group: Stable sequence or source-video identifier.
        seed: Experiment seed used to choose the ten-percent validation group.
    Returns:
        "train" or "val"; every window from a group receives the same fold.
    """
    # 1. Hash the source identifier rather than a cropped window index.
    if not group:
        raise ValueError("A nonempty source group is required")
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()

    # 2. Use a stable ten-percent holdout across repeated runs.
    return "val" if int.from_bytes(digest[:8], "big") % 10 == 0 else "train"
