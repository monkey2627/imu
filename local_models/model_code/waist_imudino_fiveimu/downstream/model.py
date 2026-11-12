"""Extract frozen IMUDINO features and regress dense articulated SMPL poses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_imu(
    backbone: nn.Module,
    imu: torch.Tensor,
    mask: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    """Encode five ordered IMU streams with one shared frozen backbone.

    Args:
        backbone: Released IMUDINO encoder in evaluation mode.
        imu: Sensor-frame six-axis data [B,L,5,6].
        mask: Valid sample flags [B,L].
        fps: Positive sample frequency in Hz.
    Returns:
        Fused patch descriptors [B,L/16,5*D], concatenated in sensor order.
    """
    # 1. Validate the shared temporal and sensor contract.
    if imu.ndim != 4 or imu.shape[2:] != (5, 6) or mask.shape != imu.shape[:2]:
        raise ValueError("Expected IMU [B,L,5,6] and mask [B,L]")
    if imu.shape[1] < 16 or imu.shape[1] % 16 or fps <= 0:
        raise ValueError("Window length must be a positive multiple of 16 and fps positive")
    batch, frames, sensors, channels = imu.shape

    # 2. Encode each physical sensor independently with the same checkpoint.
    sensor_streams = imu.permute(0, 2, 1, 3).reshape(batch * sensors, frames, channels)
    sensor_masks = mask[:, None, :].expand(batch, sensors, frames).reshape(batch * sensors, frames)
    frequencies = imu.new_full((batch * sensors,), float(fps))
    with torch.inference_mode():
        tokens = backbone({"imu_data": sensor_streams, "freqs": frequencies, "masks": sensor_masks})["patch_tokens"]

    # 3. Concatenate simultaneous patch descriptors in the fixed mounting order.
    patches, feature_dim = tokens.shape[1:]
    return tokens.reshape(batch, sensors, patches, feature_dim).permute(0, 2, 1, 3).reshape(
        batch, patches, sensors * feature_dim
    )


class FiveIMUHead(nn.Module):
    """Predict per-frame local rotations for 23 non-root SMPL joints."""

    def __init__(
        self,
        feature_dim: int = 5 * 768,
        hidden_dim: int = 512,
        output_dim: int = 23 * 6,
    ) -> None:
        """Create a projection, two-layer probe, and dense rotation head.

        Args:
            feature_dim: Concatenated sensor feature width.
            hidden_dim: Probe width after sensor fusion.
            output_dim: Six-dimensional rotation output for articulated joints.
        """
        super().__init__()
        # The projection learns cross-sensor relationships while the backbone stays fixed.
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.probe = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.frame_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor, frames: int) -> torch.Tensor:
        """Upsample patch features and predict one articulated pose per input frame.

        Args:
            features: Fused temporal descriptors [B,P,5*D].
            frames: Number of requested output frames, normally 480.
        Returns:
            Local rotation-6D predictions [B,frames,138].
        """
        # 1. Fuse the five sensor descriptors at each temporal patch.
        if features.ndim != 3 or frames < 1:
            raise ValueError("Expected patch features [B,P,D] and positive frame count")
        hidden = self.probe(self.projection(features.float()))

        # 2. Interpolate patch predictions to frame resolution before pose regression.
        dense = F.interpolate(hidden.transpose(1, 2), size=frames, mode="linear", align_corners=False)
        return self.frame_head(dense.transpose(1, 2))
