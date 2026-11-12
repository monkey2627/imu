"""Runtime MobilePose network used by the realtime Python bridge.

Inference-only re-implementation of the SparseIMU baseline cascade
(Joints 60 -> 72, Poser 72 + 60 -> 96 reduced global 6D rotations) with the
online window of the ``pnp5_60a`` models (see ``runtime_contract``).

``anchor=True`` is the identity-anchored ("sc", ``MOBILEPOSER_SHORTCUT=1``)
variant: the bones that carry a sensor output ``R_measured @ R(delta)``, where
``delta`` is the network's (small) 6D correction added to the identity 6D.
It must only be used with checkpoints that were trained that way (the
checkpoint ``manifest.json`` records it).
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from runtime_contract import (
    ANCHOR_SENSOR_SLOTS,
    OUTPUT_INDEX,
    REALTIME_LABELS,
    REALTIME_LAYOUT,
    WINDOW,
)

ROOT = Path(__file__).resolve().parents[1]
PARENTS = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
REDUCED = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
IGNORED = [7, 8, 10, 11, 20, 21, 22, 23]
LAYOUTS = {REALTIME_LAYOUT: list(REALTIME_LABELS)}


class DenseRnn(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear1 = nn.Linear(inputs, 256)
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True)
        self.linear2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = torch.relu(self.linear1(x)).transpose(0, 1)
        x, _ = self.rnn(x)
        return self.linear2(x.transpose(0, 1))


def _gram_schmidt(six):
    """(..., 6) -> (..., 3, 3) rotation whose first two columns follow the 6D input."""
    c0 = six[..., :3] / torch.linalg.vector_norm(six[..., :3], dim=-1, keepdim=True)
    c1 = six[..., 3:] - (c0 * six[..., 3:]).sum(-1, keepdim=True) * c0
    c1 = c1 / torch.linalg.vector_norm(c1, dim=-1, keepdim=True)
    return torch.stack([c0, c1, torch.cross(c0, c1, dim=-1)], dim=-1)


class MobilePose(nn.Module):
    """Inference-only wrapper for the combined 60D-to-SMPL checkpoint."""

    def __init__(self, anchor=False):
        super().__init__()
        self.anchor = bool(anchor)
        self.joints = DenseRnn(60, 72)
        self.pose = DenseRnn(132, 96)
        self.register_buffer('identity', torch.eye(3))
        self.register_buffer('identity6', torch.tensor([1., 0., 0., 0., 1., 0.]))

    def load_combined(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        for module, prefix in [(self.joints, 'joints.joints.'), (self.pose, 'pose.pose.')]:
            module.load_state_dict(
                {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)},
                strict=True,
            )
        return self.eval()

    def forward(self, imu):
        """imu: (1, WINDOW, 60) -> (24, 3, 3) local rotations of window frame OUTPUT_INDEX."""
        joints = self.joints(imu)
        self._last_joints = joints
        six = self.pose(torch.cat([joints, imu], dim=-1))[0, OUTPUT_INDEX].reshape(16, 6)
        if self.anchor:
            rot = self._anchored_rotations(six, imu[0, OUTPUT_INDEX, 15:].reshape(5, 3, 3))
        else:
            rot = _gram_schmidt(six)
        rot = torch.where(torch.isnan(rot), torch.zeros_like(rot), rot)
        global_rot = [rot[REDUCED.index(j)] if j in REDUCED else self.identity for j in range(24)]
        local = [global_rot[0]]
        for j in range(1, 24):
            local.append(self.identity if j in IGNORED else global_rot[PARENTS[j]].T @ global_rot[j])
        return torch.stack(local)

    def _anchored_rotations(self, six, measured):
        rot = list(_gram_schmidt(six).unbind(0))
        for joint, slot in ANCHOR_SENSOR_SLOTS.items():
            k = REDUCED.index(joint)
            r_meas = measured[slot]
            if float(r_meas.abs().sum()) <= 0.5:      # zero-padded step: identity (as in training)
                r_meas = self.identity
            rot[k] = r_meas @ _gram_schmidt(six[k] + self.identity6)
        return torch.stack(rot)


def windows(frames):
    window = None
    for frame in frames:
        window = np.repeat(frame[None], WINDOW, axis=0) if window is None else np.concatenate([window[1:], frame[None]])
        yield window[None].astype(np.float32)
