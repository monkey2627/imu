"""Part-wise discrete pose tokens and the token-decoding pose head.

Tokenizer: one residual-VQ autoencoder per body part (trunk / arms / legs) on parent-relative rotations (6D). The root
(pelvis) orientation is not tokenized. Decoded parts are composed along the SMPL chain with the root to give the global
rotations of the 16 reduced joints (Poser's output space).

TokenPoseNet replaces Poser.pose when MOBILEPOSER_TOKEN=1: a (bi)LSTM backbone predicts, per frame (or per T-frame segment),
a categorical distribution over every part's code at every residual level (+ the pelvis 6D); the frozen decoders turn the
codes into rotations. See the class docstring for the decoding modes.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mobileposer.articulate.math import r6d_to_rotation_matrix, rotation_matrix_to_r6d
from mobileposer.models.rnn import RNN

PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
REDUCED = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
PARTS = {                                   # SMPL joint ids, parent-before-child order; root (0) is never tokenized
    'trunk': [3, 6, 9, 12, 15],
    'l_arm': [13, 16, 18],
    'r_arm': [14, 17, 19],
    'l_leg': [1, 4],
    'r_leg': [2, 5],
}


class VQ(nn.Module):
    def __init__(self, K, dim, decay=0.99):
        super().__init__()
        self.K, self.decay = K, decay
        self.register_buffer('emb', torch.randn(K, dim) * 0.1)
        self.register_buffer('cnt', torch.ones(K))
        self.register_buffer('sum', self.emb.clone())
        self.register_buffer('inited', torch.tensor(0))

    def forward(self, z):
        if self.training and self.inited.item() == 0:                          # init codebook from data
            self.emb.copy_(z[torch.randperm(len(z), device=z.device)[:self.K]].detach())
            self.sum.copy_(self.emb)
            self.inited.fill_(1)
        d = (z ** 2).sum(1, keepdim=True) - 2 * z @ self.emb.t() + (self.emb ** 2).sum(1)[None]
        idx = d.argmin(1)
        q = self.emb[idx]
        if self.training:
            with torch.no_grad():
                oh = F.one_hot(idx, self.K).float()
                self.cnt.mul_(self.decay).add_(oh.sum(0), alpha=1 - self.decay)
                self.sum.mul_(self.decay).add_(oh.t() @ z.detach(), alpha=1 - self.decay)
                self.emb.copy_(self.sum / self.cnt.clamp(min=1e-5)[:, None])
                dead = self.cnt < 1e-3                                         # re-seed dead codes
                if dead.any():
                    self.emb[dead] = z[torch.randint(len(z), (int(dead.sum()),), device=z.device)].detach()
                    self.cnt[dead] = 1.0
                    self.sum[dead] = self.emb[dead]
        commit = F.mse_loss(z, q.detach())
        return z + (q - z).detach(), commit, idx


class PartAE(nn.Module):
    def __init__(self, n_joint, K, dim=64, hid=512, nq=1):
        super().__init__()
        self.nq = nq                                                            # residual VQ levels (1 = plain VQ)
        self.enc = nn.Sequential(nn.Linear(n_joint * 6, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, dim))
        self.dec = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, n_joint * 6))
        self.vq = nn.ModuleList([VQ(K, dim) for _ in range(nq)]) if K > 0 else None

    def forward(self, x):
        z = self.enc(x)
        commit, idxs = 0.0, []
        if self.vq is not None:
            res, zq = z, 0
            for vq in self.vq:
                q, c, idx = vq(res)
                zq = zq + q
                res = res - q.detach()
                commit = commit + c
                idxs.append(idx)
            z = zq
        return self.dec(z), commit, idxs


def global_to_local(glob):
    """glob [N,24,3,3] -> parent-relative rotations [N,24,3,3] (root keeps its global)."""
    loc = [glob[:, 0]] + [glob[:, PARENT[j]].transpose(-1, -2) @ glob[:, j] for j in range(1, 24)]
    return torch.stack(loc, 1)


class TokenPoseNet(nn.Module):
    """Drop-in for Poser.pose: forward(x [B,S,F], lengths) -> ([B,S,96] global reduced 6D, None, None).

    MOBILEPOSER_TOKEN_MODE:
      soft    (default) decode the expected code embedding (differentiable); inference soft, or argmax with TOKEN_INFER=hard
      gumbel  discrete sampling: straight-through Gumbel-softmax code per residual level in training, arg-max at inference
      ar      coarse-to-fine autoregressive over the residual levels: level l is predicted from the backbone feature and the
              (teacher-forced GT / predicted) codes of the levels < l; Gumbel straight-through decode in training, sequential arg-max at inference
    MOBILEPOSER_TOKEN_SEG=T: segment tokens, one code set per T consecutive frames (tokenizer must be trained with the same T).
    """

    def __init__(self, n_input, n_hidden=256, K=1024, nq=3, tokenizer_path=None, dim=64, hid=512):
        super().__init__()
        self.K, self.nq, self.dim = K, nq, dim
        self.T = int(os.environ.get('MOBILEPOSER_TOKEN_SEG', 1))
        self.mode = os.environ.get('MOBILEPOSER_TOKEN_MODE', 'soft')
        self.tau = float(os.environ.get('MOBILEPOSER_TOKEN_TAU', 1.0))
        self.backbone = RNN(n_input, n_hidden, n_hidden)
        self.act = nn.GELU()
        if self.T > 1:
            self.seg = nn.Sequential(nn.Linear(self.T * n_hidden, n_hidden), nn.GELU())
        if self.mode == 'ar':
            self.heads_ar = nn.ModuleDict({n: nn.ModuleList([nn.Linear(n_hidden + dim, K) for _ in range(nq)]) for n in PARTS})
        else:
            self.heads = nn.ModuleDict({n: nn.Linear(n_hidden, nq * K) for n in PARTS})
        self.pelvis = nn.Linear(n_hidden, 6)
        if self.mode == 'aux':
            self.out_head = nn.Linear(n_hidden, len(REDUCED) * 6)
        self.tok = nn.ModuleDict({n: PartAE(len(j) * self.T, K, dim=dim, hid=hid, nq=nq) for n, j in PARTS.items()})
        tokenizer_path = tokenizer_path or os.environ.get('MOBILEPOSER_TOKENIZER')
        if tokenizer_path:
            ck = torch.load(tokenizer_path, map_location='cpu')
            for n in PARTS:
                self.tok[n].load_state_dict(ck['state'][n])
        for p in self.tok.parameters():
            p.requires_grad_(False)
        self.use_res = os.environ.get('MOBILEPOSER_TOKEN_RES') == '1'
        if self.use_res:                                                        # continuous correction of the decoded 6D, zero-initialised
            self.res = nn.ModuleDict({n: nn.Sequential(nn.Linear(n_hidden + dim, 256), nn.GELU(), nn.Linear(256, len(j) * 6)) for n, j in PARTS.items()})
            for m in self.res.values():
                nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        self.infer = os.environ.get('MOBILEPOSER_TOKEN_INFER', 'soft')
        self.last_logits = None
        self.tf_targets = None                                                  # set by Poser.shared_step (teacher forcing for mode ar)
        self._gt = None

    def train(self, mode=True):
        super().train(mode)
        self.tok.eval()                                                         # tokenizer is frozen: no EMA updates
        return self

    def _code_z(self, name, fs, hard):
        """Per-part code distributions -> summed code embedding [Ns,dim]; stores logits [Ns,nq,K]."""
        emb = [vq.emb for vq in self.tok[name].vq]
        if self.mode == 'ar':
            zc, zd, lgs = None, 0, []
            gt = self._gt[name] if (self.training and self._gt is not None) else None
            for l in range(self.nq):
                zin = zc if zc is not None else fs.new_zeros(len(fs), self.dim)
                lg = self.heads_ar[name][l](torch.cat([fs, zin], -1))
                lgs.append(lg)
                if gt is not None:                                              # teacher forcing: condition on GT codes, decode the sampled ones
                    e = emb[l][gt[:, l]]
                    zd = zd + F.gumbel_softmax(lg, tau=self.tau, hard=True) @ emb[l]
                else:
                    e = emb[l][lg.argmax(-1)]
                    zd = zd + e
                zc = e if zc is None else zc + e
            self.last_logits[name] = torch.stack(lgs, 1)
            return zd
        lg = self.heads[name](fs).view(-1, self.nq, self.K)
        self.last_logits[name] = lg
        z = 0
        for l in range(self.nq):
            if self.mode == 'gumbel' and self.training:
                z = z + F.gumbel_softmax(lg[:, l], tau=self.tau, hard=True) @ emb[l]
            elif self.mode == 'gumbel' or hard:
                z = z + emb[l][lg[:, l].argmax(-1)]
            else:
                z = z + lg[:, l].softmax(-1) @ emb[l]
        return z

    def forward(self, x, seq_lengths=None, h=None):
        B, S, _ = x.shape
        T = self.T
        f = self.act(self.backbone(x, seq_lengths)[0])
        if f.shape[1] < S:                                                      # pad_packed_sequence trims to the longest sequence
            f = F.pad(f, (0, 0, 0, S - f.shape[1]))
        Sp = -(-S // T) * T
        if Sp > S:
            f = F.pad(f, (0, 0, 0, Sp - S))
        N, G = B * Sp, Sp // T
        fr = f.reshape(N, -1)
        fs = self.seg(f.reshape(B, G, -1)).reshape(B * G, -1) if T > 1 else fr
        hard = (not self.training) and self.infer == 'hard'
        self._gt = None
        if self.mode == 'ar' and self.training and self.tf_targets is not None:
            self._gt = self.gt_tokens(self.tf_targets)
        if self.mode == 'aux':                                                  # continuous output; code prediction is only an auxiliary (cross-entropy) task
            self.last_logits = {n: self.heads[n](fs).view(-1, self.nq, self.K) for n in PARTS}
            return self.out_head(fr).view(B, Sp, -1)[:, :S].contiguous(), None, None
        self.last_logits = {}
        glob = [None] * 24
        glob[0] = r6d_to_rotation_matrix(self.pelvis(fr))                       # [N,3,3]
        local = {}
        for n, joints in PARTS.items():
            z = self._code_z(n, fs, hard)
            rec = self.tok[n].dec(z)
            if self.use_res:
                rec = rec + self.res[n](torch.cat([fs, z], -1))
            R = r6d_to_rotation_matrix(rec.reshape(-1, 6)).view(N, len(joints), 3, 3)       # frame-major within a segment: [B*G,T,nj] == [B,Sp,nj]
            for i, j in enumerate(joints):
                local[j] = R[:, i]
        for j in range(1, 24):
            if j in local:
                glob[j] = glob[PARENT[j]] @ local[j]
        out = torch.stack([rotation_matrix_to_r6d(glob[j]).view(N, 6) for j in REDUCED], 1)
        return out.view(B, Sp, -1)[:, :S].contiguous(), None, None

    @torch.no_grad()
    def gt_tokens(self, target_global_6d):
        """target [B,S,24,6] (global) -> {part: idx [B*G, nq]} from the frozen tokenizer."""
        B, S = target_global_6d.shape[:2]
        T = self.T
        Sp = -(-S // T) * T
        if Sp > S:
            target_global_6d = F.pad(target_global_6d, (0, 0, 0, 0, 0, Sp - S))
        N = B * Sp
        Rg = r6d_to_rotation_matrix(target_global_6d.reshape(-1, 6)).view(N, 24, 3, 3)
        loc = global_to_local(Rg)
        out = {}
        for n, joints in PARTS.items():
            xf = rotation_matrix_to_r6d(loc[:, joints].reshape(-1, 3, 3)).view(B * (Sp // T), -1)
            out[n] = torch.stack(self.tok[n](xf)[2], 1)
        return out

    def token_loss(self, target_global_6d, valid):
        """Cross-entropy of the predicted code distributions vs the GT tokens; valid [B,S] bool."""
        tgt = self._gt if self._gt is not None else self.gt_tokens(target_global_6d)
        B, S = valid.shape
        T = self.T
        Sp = -(-S // T) * T
        v = F.pad(valid, (0, Sp - S), value=False).view(B, Sp // T, T).all(-1).reshape(-1)
        loss = 0
        for n in PARTS:
            lg = self.last_logits[n][v]
            loss = loss + F.cross_entropy(lg.reshape(-1, self.K), tgt[n][v].reshape(-1))
        return loss / len(PARTS)
