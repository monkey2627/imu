"""KinFormer: a kinematics-aware space-time transformer for sparse-IMU pose estimation.

Tokens
  sensor tokens (5, per frame): [acc(3), ori(9)] (+ lever-arm compensated features (9) + sensor lever arm r (3) + template
                  surface coordinate (3), when enabled) + learned sensor-id embedding.  One temporal attention layer first.
  joint tokens (24, per frame): learned joint embedding + tree-depth embedding (+ noisy joint position (3) for the Poser).
  body context (optional): the 10 body-size numbers are added to every token through one linear layer.
Blocks (x L)
  spatial  : joint tokens self-attend (learned bias per SMPL tree distance and head) and cross-attend to the five sensor
             tokens (learned bias per tree distance between the joint and the sensor's bone joint and head)
  temporal : every joint token attends along time (ALiBi distance bias, bidirectional)
Output heads read the joint tokens: Poser -> 6D rotation of the 16 reduced joints; Joints -> 3D position of all 24 joints.
Input layout x [B,S,F]: [noisy joints 72 (poser only)] | acc 15 | ori 45 | context 37 (if CTX) | lever features 36 (if LEVER)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

PARENT = [None, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SENSOR_JOINT = [18, 19, 4, 5, 3]
REDUCED = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]


def tree_distance():
    D = torch.zeros(24, 24, dtype=torch.long)
    anc = []
    for i in range(24):
        a, j = [], i
        while j is not None:
            a.append(j)
            j = PARENT[j]
        anc.append(a)
    for i in range(24):
        for j in range(24):
            common = next(k for k in anc[i] if k in anc[j])
            D[i, j] = anc[i].index(common) + anc[j].index(common)
    return D


def depth():
    d = []
    for i in range(24):
        n, j = 0, i
        while PARENT[j] is not None:
            n, j = n + 1, PARENT[j]
        d.append(n)
    return torch.tensor(d)


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x, mem, bias=None, chunk=None):
        N, L, d = x.shape
        q = self.q(x).view(N, L, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(mem).view(N, mem.shape[1], 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if chunk is None or N <= chunk:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            y = torch.cat([F.scaled_dot_product_attention(q[i:i + chunk], k[i:i + chunk], v[i:i + chunk], attn_mask=bias)
                           for i in range(0, N, chunk)])
        return self.o(y.transpose(1, 2).reshape(N, L, d))


class FFN(nn.Module):
    def __init__(self, d, mult=3, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Dropout(drop), nn.Linear(mult * d, d))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.n = nn.ModuleList([nn.LayerNorm(d) for _ in range(6)])
        self.self_attn, self.cross_attn, self.ff1 = Attn(d, heads), Attn(d, heads), FFN(d, drop=drop)
        self.t_attn, self.ff2 = Attn(d, heads), FFN(d, drop=drop)
        self.drop = nn.Dropout(drop)

    def forward(self, J, S_tok, b_self, b_cross, b_time, B, T):
        # J [B*T,24,d]; S_tok [B*T,5,d]
        x = self.n[0](J)
        J = J + self.drop(self.self_attn(x, x, b_self))
        J = J + self.drop(self.cross_attn(self.n[1](J), S_tok, b_cross))
        J = J + self.drop(self.ff1(self.n[2](J)))
        d = J.shape[-1]
        Jt = J.view(B, T, 24, d).transpose(1, 2).reshape(B * 24, T, d)
        x = self.n[3](Jt)
        chunk = max(1, int(1.5e9 // (b_time.shape[1] * T * T))) if T > 400 else None
        Jt = Jt + self.drop(self.t_attn(x, x, b_time, chunk=chunk))
        Jt = Jt + self.drop(self.ff2(self.n[4](Jt)))
        return Jt.view(B, 24, T, d).transpose(1, 2).reshape(B * T, 24, d)


class KinFormer(nn.Module):
    def __init__(self, mode, d=128, layers=3, heads=4, drop=0.1):
        super().__init__()
        self.mode, self.d, self.heads = mode, d, heads
        self.has_ctx = os.environ.get('MOBILEPOSER_CTX') == '1'
        self.has_lever = os.environ.get('MOBILEPOSER_LEVER') in ('1', 'arms')
        use = os.environ.get('MOBILEPOSER_CTX_USE', 'r,len,pos').split(',')
        self.use_r, self.use_len, self.use_pos = 'r' in use, 'len' in use, 'pos' in use
        s_in = 12 + (9 if self.has_lever else 0) + (3 if self.use_r else 0) + (3 if self.use_pos else 0)
        self.sensor_in = nn.Linear(s_in, d)
        self.sensor_id = nn.Embedding(5, d)
        self.sensor_t = nn.ModuleList([nn.LayerNorm(d), Attn(d, heads), FFN(d, drop=drop)])
        self.joint_emb, self.depth_emb = nn.Embedding(24, d), nn.Embedding(10, d)
        self.pos_in = nn.Linear(3, d) if mode == 'poser' else None
        self.len_in = nn.Linear(10, d) if (self.has_ctx and self.use_len) else None
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 6 if mode == 'poser' else 3)
        D = tree_distance()
        self.register_buffer('D_self', D, persistent=False)
        self.register_buffer('D_cross', D[:, SENSOR_JOINT], persistent=False)
        self.register_buffer('depth', depth(), persistent=False)
        self.bias_self = nn.Embedding(int(D.max()) + 1, heads)
        self.bias_cross = nn.Embedding(int(D.max()) + 1, heads)
        nn.init.zeros_(self.bias_self.weight), nn.init.zeros_(self.bias_cross.weight)
        self.register_buffer('slopes', torch.tensor([2 ** (-8.0 * (i + 1) / heads) for i in range(heads)]), persistent=False)
        self.register_buffer('reduced', torch.tensor(REDUCED), persistent=False)

    def parse(self, x):
        B, T, _ = x.shape
        o, jpos = 0, None
        if self.mode == 'poser':
            jpos, o = x[..., :72].reshape(B, T, 24, 3), 72
        acc = x[..., o:o + 15].reshape(B, T, 5, 3)
        ori = x[..., o + 15:o + 60].reshape(B, T, 5, 9)
        o += 60
        ctx = None
        if self.has_ctx:
            ctx, o = x[..., o:o + 37], o + 37
        lev = x[..., o:o + 36].reshape(B, T, 3, 4, 3) if self.has_lever else None
        return jpos, acc, ori, ctx, lev

    def forward(self, x, seq_lengths=None, h=None):
        B, T, _ = x.shape
        jpos, acc, ori, ctx, lev = self.parse(x)
        feats = [acc, ori]
        if self.has_lever:
            lf = torch.cat([lev[:, :, 0], lev[:, :, 1], lev[:, :, 2]], dim=-1)                  # [B,T,4,9]
            feats.append(F.pad(lf, (0, 0, 0, 1)))                                                 # waist: zeros
        if self.has_ctx and self.use_r:
            feats.append(F.pad(ctx[..., :12].reshape(B, T, 4, 3), (0, 0, 0, 1)))
        if self.has_ctx and self.use_pos:
            feats.append(ctx[..., 22:37].reshape(B, T, 5, 3))
        S_tok = self.sensor_in(torch.cat(feats, dim=-1)) + self.sensor_id.weight                  # [B,T,5,d]
        gl = self.len_in(ctx[..., 12:22]) if self.len_in is not None else None                    # [B,T,d]
        if gl is not None:
            S_tok = S_tok + gl.unsqueeze(2)
        d = self.d
        pos = torch.arange(T, device=x.device)
        b_time = (-self.slopes.view(1, -1, 1, 1) * (pos[None, :] - pos[:, None]).abs().float().view(1, 1, T, T))   # [1,H,T,T]
        st = S_tok.transpose(1, 2).reshape(B * 5, T, d)
        n, at, ff = self.sensor_t
        chunk = max(1, int(1.5e9 // (self.heads * T * T))) if T > 400 else None
        st = st + at(n(st), n(st), b_time, chunk=chunk)
        st = st + ff(st)
        S_tok = st.view(B, 5, T, d).transpose(1, 2).reshape(B * T, 5, d)
        J = (self.joint_emb.weight + self.depth_emb(self.depth)).view(1, 1, 24, d).expand(B, T, 24, d)
        if jpos is not None:
            J = J + self.pos_in(jpos)
        if gl is not None:
            J = J + gl.unsqueeze(2)
        J = J.reshape(B * T, 24, d)
        b_self = self.bias_self(self.D_self).permute(2, 0, 1).unsqueeze(0)                         # [1,H,24,24]
        b_cross = self.bias_cross(self.D_cross).permute(2, 0, 1).unsqueeze(0)                      # [1,H,24,5]
        for blk in self.blocks:
            if self.training and os.environ.get("MOBILEPOSER_CKPT", "0") == "1":
                J = torch.utils.checkpoint.checkpoint(blk, J, S_tok, b_self, b_cross, b_time, B, T, use_reentrant=False)
            else:
                J = blk(J, S_tok, b_self, b_cross, b_time, B, T)
        J = self.norm(J).view(B, T, 24, d)
        if self.mode == 'poser':
            out = self.out(J[:, :, self.reduced]).reshape(B, T, -1)
        else:
            out = self.out(J).reshape(B, T, -1)
        return out, None, None
