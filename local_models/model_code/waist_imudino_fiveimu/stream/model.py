"""Stateful causal head: slow IMUDINO tokens + fast per-frame sensors -> local 6D per frame."""
import torch
from torch import nn

from .common import FRAME_DIM, TOKEN_DIM


class StreamHead(nn.Module):
    def __init__(self, token_dim=TOKEN_DIM, frame_dim=FRAME_DIM, token_hidden=256, frame_hidden=128,
                 hidden=512, layers=2, output_dim=23 * 6, dropout=0.1):
        super().__init__()
        self.token_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Dropout(dropout),
                                        nn.Linear(token_dim, token_hidden), nn.GELU())
        self.frame_proj = nn.Sequential(nn.Linear(frame_dim, frame_hidden), nn.GELU())
        self.gru = nn.GRU(token_hidden + frame_hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, tokens, frames, state=None):
        """tokens [B,T,TOKEN_DIM] (newest usable token per frame, zeros if none), frames [B,T,FRAME_DIM]."""
        x = torch.cat((self.token_proj(tokens.float()), self.frame_proj(frames.float())), dim=-1)
        y, state = self.gru(x, state)
        return self.out(y), state
