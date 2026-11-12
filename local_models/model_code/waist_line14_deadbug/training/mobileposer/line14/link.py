"""Shared C20 sensor-to-bone link encoder used by training and calibration inference."""
from __future__ import annotations

import torch
from torch import nn

from mobileposer.line9.enc2 import FeatEncoder
from mobileposer.line9.calibration_encoder import rotation_6d_to_matrix


class LinkEncoder(FeatEncoder):
    def __init__(self, in_dim, geo_dim=24, with_rot=True, with_geo=True, **kw):
        super().__init__(in_dim, **kw)
        del self.head
        latent = self.norm.normalized_shape[0]
        self.with_rot, self.with_geo, self.geo_dim = with_rot, with_geo, geo_dim
        if with_geo:
            self.geo = nn.Linear(latent, 2 * geo_dim)
            nn.init.zeros_(self.geo.weight)
            nn.init.zeros_(self.geo.bias)
        if with_rot:
            self.rot = nn.Linear(latent, 30)
            nn.init.zeros_(self.rot.weight)
            with torch.no_grad():
                self.rot.bias.copy_(torch.tensor([1.0, 0, 0, 0, 1.0, 0]).repeat(5))

    def latent(self, x, phase_ids, mask):
        f = self.proj(x) + self.phase_emb(phase_ids.masked_fill(~mask, 0))
        enc, _ = self.gru(f)
        safe = phase_ids.masked_fill(~mask, 0)
        member = (safe[:, None, :] == torch.arange(self.max_phases, device=x.device)[None, :, None]) & mask[:, None, :]
        pm = member.any(-1)
        score = self.frame_att(enc).squeeze(-1)[:, None, :].masked_fill(~member, -torch.inf).masked_fill(~pm[:, :, None], 0.0)
        w = torch.softmax(score, -1) * member.to(score.dtype)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        tokens = self.phase_proj(torch.einsum("bpt,btd->bpd", w, enc))
        pooled, _ = self.att(self.query.expand(x.shape[0], -1, -1), tokens, tokens, key_padding_mask=~pm, need_weights=False)
        return self.norm(pooled[:, 0])

    def forward(self, x, phase_ids, mask):
        z = self.latent(x, phase_ids, mask)
        geo = self.geo(z) if self.with_geo else None
        rotation = rotation_6d_to_matrix(self.rot(z).reshape(-1, 5, 6)) if self.with_rot else None
        return geo, rotation
