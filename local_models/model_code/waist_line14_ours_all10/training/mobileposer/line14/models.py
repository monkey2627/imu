"""Joints regressors with richer geometry conditioning for line 14.

``FilmJoints`` keeps the public Joints backbone (Linear(60,256) -> ReLU/Dropout -> 2-layer BiLSTM(256) -> Linear(512,72)) and
conditions on the (standardised) geometry vector c at two places, FiLM style, both zero-initialised so training starts at B0:
  input stage : h <- h * (1 + gamma_in(c)) + beta_in(c)        (256 channels, before ReLU)
  output stage: y <- y * (1 + gamma_out(c)) + beta_out(c)      (72 outputs; joint positions scale with body size)
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mobileposer.line9.models import Line9Joints


class FilmJoints(Line9Joints):
    def __init__(self, geom_dim, geometry_stats):
        super().__init__(mode="oracle", geom_dim=geom_dim, geometry_stats=geometry_stats)
        del self.geometry_projection                  # replaced by the FiLM layers
        self.geometry_projection = None
        self.film_in = nn.Linear(geom_dim, 2 * self.hidden_dim)
        self.film_out = nn.Linear(geom_dim, 2 * self.output_dim)
        for m in (self.film_in, self.film_out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, imu, input_lengths=None, geometry=None):
        c = self._conditioning(imu, geometry)
        data = self.linear1(imu)
        g, b = self.film_in(c).chunk(2, dim=-1)
        data = data * (1.0 + g[:, None, :]) + b[:, None, :]
        data = self.dropout(F.relu(data))
        if input_lengths is not None:
            lengths = torch.as_tensor(input_lengths, dtype=torch.long, device="cpu")
            data = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        data, _ = self.rnn(data)
        if input_lengths is not None:
            data, _ = pad_packed_sequence(data, batch_first=True)
        out = self.linear2(data)
        go, bo = self.film_out(c).chunk(2, dim=-1)
        return out * (1.0 + go[:, None, :]) + bo[:, None, :]
