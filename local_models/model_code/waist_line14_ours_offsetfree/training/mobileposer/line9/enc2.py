"""Generic I2S-rotation encoder (any input feature dimension) and physics-motivated sensor-frame features.

Same structure as ``CalibrationEncoder`` (frame projection + phase embedding + BiGRU + per-phase attention pooling +
persistent query), rotation head only.  ``phys_features`` expresses the data in each sensor's own frame (gravity direction,
acceleration, angular velocity) and relative to the pelvis sensor: these are the quantities that constrain R_off directly
(a hinge axis is fixed in the segment frame; gravity fixes tilt) and are invariant to heading.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mobileposer.line9.calibration_encoder import rotation_6d_to_matrix


class FeatEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2, latent: int = 256, heads: int = 4,
                 max_phases: int = 8, dropout: float = 0.1):
        super().__init__()
        self.max_phases = max_phases
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.phase_emb = nn.Embedding(max_phases, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.frame_att = nn.Linear(2 * hidden, 1)
        self.phase_proj = nn.Linear(2 * hidden, latent)
        self.query = nn.Parameter(torch.randn(1, 1, latent) * 0.02)
        self.att = nn.MultiheadAttention(latent, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(latent)
        self.head = nn.Linear(latent, 5 * 6)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.0, 0, 0, 0, 1.0, 0]).repeat(5))

    def forward(self, x, phase_ids, mask):
        safe = phase_ids.masked_fill(~mask, 0)
        f = self.proj(x) + self.phase_emb(safe)
        lengths = mask.sum(1).cpu()
        enc, _ = self.gru(pack_padded_sequence(f, lengths, batch_first=True, enforce_sorted=False))
        enc, _ = pad_packed_sequence(enc, batch_first=True, total_length=x.shape[1])
        member = (safe[:, None, :] == torch.arange(self.max_phases, device=x.device)[None, :, None]) & mask[:, None, :]
        pm = member.any(-1)
        score = self.frame_att(enc).squeeze(-1)[:, None, :].masked_fill(~member, -torch.inf).masked_fill(~pm[:, :, None], 0.0)
        w = torch.softmax(score, -1) * member.to(score.dtype)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        tokens = self.phase_proj(torch.einsum("bpt,btd->bpd", w, enc))
        pooled, _ = self.att(self.query.expand(x.shape[0], -1, -1), tokens, tokens, key_padding_mask=~pm, need_weights=False)
        return rotation_6d_to_matrix(self.head(self.norm(pooled[:, 0])).reshape(-1, 5, 6))


def vee_skew(M: torch.Tensor) -> torch.Tensor:
    S = 0.5 * (M - M.transpose(-1, -2))
    return torch.stack((S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]), -1)


def phys_features(imu: torch.Tensor, dt: float = 4.0 / 60.0) -> torch.Tensor:
    """imu [...,T,60] (heading-normalized, acc/30 | ori) -> [...,T,135] sensor-frame features."""
    lead = imu.shape[:-1]
    acc = imu[..., :15].reshape(*lead, 5, 3)
    R = imu[..., 15:].reshape(*lead, 5, 3, 3)
    g_s = R[..., 1, :]                                            # world vertical in the sensor frame
    a_s = (R.transpose(-1, -2) @ acc[..., None]).squeeze(-1)      # acceleration in the sensor frame
    dR = R[..., :-1, :, :, :].transpose(-1, -2) @ R[..., 1:, :, :, :]
    w = vee_skew(dR) / dt                                         # gyro-like angular velocity in the sensor frame
    w = torch.cat((w, w[..., -1:, :, :]), dim=-3)
    rel = R[..., 4:5, :, :].transpose(-1, -2) @ R                 # orientation relative to the pelvis sensor
    return torch.cat((g_s, a_s, w, R.reshape(*lead, 5, 9), rel.reshape(*lead, 5, 9)), -1).reshape(*lead, 135)
