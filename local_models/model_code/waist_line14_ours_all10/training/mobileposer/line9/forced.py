"""Forced-bone-length control for line 9.

The backbone (identical to the public-Joints replica, i.e. Line9Joints mode "b0")
emits 24 vectors: vector 0 is a root-position residual, vectors 1..23 are bone
directions (normalised) in the global frame.  Joint positions are rebuilt along
the SMPL tree as ``pos[j] = pos[parent[j]] + dir[j] * length[j]``.  ``fb0`` uses
training-identity mean lengths/root; ``foracle`` uses the true ones.  Parameter
count is identical, so the only difference is geometry entering the output.
"""
from __future__ import annotations

import torch
from torch import nn

from mobileposer.line9.models import Line9Joints


@torch.no_grad()
def build_table(dataset, records, train_identities):
    body = dataset._body()
    ident = {}
    for r in records:
        if r.identity in ident:
            continue
        beta, lengths = dataset._get_geometry(r)
        root = body.get_zero_pose_joint_and_vertex(beta)[0].reshape(24, 3)[0]
        ident[r.identity] = torch.cat((lengths[1:], root))  # 23 + 3
    train = torch.stack([v for k, v in ident.items() if k in train_identities])
    return {"ident": ident, "mean_lengths": train[:, :23].mean(0), "mean_root": train[:, 23:].mean(0)}


def make_G(table):
    def G(b):
        return torch.stack([table["ident"][i] for i in b["identity"]])
    return G


class ForcedBoneModel(nn.Module):
    def __init__(self, mode, mean_lengths, mean_root, parent):
        super().__init__()
        assert mode in ("fb0", "foracle")
        self.mode = mode
        self.backbone = Line9Joints(mode="b0", geom_dim=33)
        self.parent = [-1 if p is None else int(p) for p in parent]
        assert all(self.parent[j] < j for j in range(1, 24))
        self.register_buffer("mean_lengths", mean_lengths.clone())
        self.register_buffer("mean_root", mean_root.clone())

    def forward(self, imu, input_lengths=None, geom=None):
        raw = self.backbone(imu, input_lengths).float().reshape(imu.shape[0], imu.shape[1], 24, 3)
        if self.mode == "foracle":
            lengths, root = geom[:, :23].float(), geom[:, 23:].float()
        else:
            lengths = self.mean_lengths[None].expand(imu.shape[0], -1)
            root = self.mean_root[None].expand(imu.shape[0], -1)
        direction = torch.nn.functional.normalize(raw[:, :, 1:], dim=-1, eps=1e-6)
        pos = [root[:, None, :] + raw[:, :, 0]]
        for j in range(1, 24):
            pos.append(pos[self.parent[j]] + direction[:, :, j - 1] * lengths[:, None, j - 1, None])
        return torch.stack(pos, dim=2).reshape(imu.shape[0], imu.shape[1], 72)
