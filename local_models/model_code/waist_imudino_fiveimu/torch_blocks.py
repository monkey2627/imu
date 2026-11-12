"""Evaluation-only timm 1.0.30 operator equivalents for this released backbone.

Matches its default LayerNorm eps=1e-6, GELU MLP and unbiassed QKV attention.
No timm/torchvision installation or changes to the existing environment needed.
This deliberately does not implement training-time stochastic depth.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(dim, eps=1e-6)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.num_heads,
                                     dim // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).reshape(batch, length, dim))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, act_layer,
                 norm_layer, drop_path, mlp_layer):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(dim * mlp_ratio), act_layer)

    def forward(self, x, attn_mask=None):
        if self.training:
            raise RuntimeError("Portable IMUDINO blocks support evaluation only; train in the server repository")
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        return x + self.mlp(self.norm2(x))
