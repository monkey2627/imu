"""Sequence-model factory for Poser / Joints.

MOBILEPOSER_ARCH:
  lstm     (default) the original 2-layer bidirectional LSTM, hidden 256
  biglstm  3-layer bidirectional LSTM, hidden 512  (capacity control)
  tf       Transformer encoder with ALiBi distance bias + a depthwise-conv front end (local dynamics), bidirectional,
           d_model MOBILEPOSER_TF_DIM (384), layers MOBILEPOSER_TF_LAYERS (6), heads 6.  Drop-in replacement for
           models.rnn.RNN: forward(x [B,S,F], seq_lengths) -> (out [B,S,n_out], None, None).
"""
import math
import os

import torch
import torch.nn as nn

if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)   # the eval fast path returns NaN on padded rows (breaks validation loss)

from mobileposer.models.rnn import RNN


def arch():
    return os.environ.get('MOBILEPOSER_ARCH', 'lstm')


class TransformerSeq(nn.Module):
    def __init__(self, n_input, n_output, d_model=384, n_layer=6, n_head=6, ff=1024, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.inp = nn.Sequential(nn.Linear(n_input, d_model), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)   # local temporal context (acc derivatives)
        self.mix = nn.Linear(d_model, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, ff, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_output)
        slopes = torch.tensor([2 ** (-8.0 * (i + 1) / n_head) for i in range(n_head)])
        self.register_buffer('slopes', slopes, persistent=False)

    def forward(self, x, seq_lengths=None, h=None):
        B, S, _ = x.shape
        z = self.inp(x)
        z = z + self.mix(torch.nn.functional.gelu(self.conv(z.transpose(1, 2)).transpose(1, 2)))
        pos = torch.arange(S, device=x.device)
        dist = (pos[None, :] - pos[:, None]).abs().float()                                      # [S,S]
        bias = -self.slopes.view(-1, 1, 1) * dist.unsqueeze(0)                                    # [H,S,S]
        bias = bias.unsqueeze(0).expand(B, -1, -1, -1)                                            # [B,H,S,S]
        if seq_lengths is not None:
            lens = torch.as_tensor(seq_lengths, device=x.device)
            pad = (pos[None, :] >= lens[:, None])                                                 # [B,S] True = padded key
            bias = bias.masked_fill(pad[:, None, None, :], float('-inf'))
        mask = bias.reshape(B * self.n_head, S, S)
        z = self.enc(z, mask=mask)
        return self.out(self.norm(z)), None, None


def make_seq_model(n_input, n_output, n_hidden=256, mode=None, **kw):
    a = arch()
    if a == "kin":
        from mobileposer.models.kinformer import KinFormer
        return KinFormer(mode, d=int(os.environ.get("MOBILEPOSER_KIN_DIM", 128)), layers=int(os.environ.get("MOBILEPOSER_KIN_LAYERS", 3)), heads=int(os.environ.get("MOBILEPOSER_KIN_HEADS", 4)))
    if a == 'tf':
        return TransformerSeq(n_input, n_output, d_model=int(os.environ.get('MOBILEPOSER_TF_DIM', 384)),
                              n_layer=int(os.environ.get('MOBILEPOSER_TF_LAYERS', 6)))
    if a == 'biglstm':
        return RNN(n_input, n_output, 512, n_rnn_layer=3, **kw)
    return RNN(n_input, n_output, n_hidden, **kw)
