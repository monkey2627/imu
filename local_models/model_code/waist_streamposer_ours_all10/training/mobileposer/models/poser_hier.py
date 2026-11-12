import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mobileposer.config import *
import mobileposer.articulate as art
from mobileposer.models.poser import Poser

SMPL_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
IDENTITY_6D = [1., 0., 0., 0., 1., 0.]


def _parent_idx():
    red = list(joint_set.reduced)
    idx = {j: i for i, j in enumerate(red)}
    return [idx[SMPL_PARENT[j]] if SMPL_PARENT[j] in idx else -1 for j in red]


class HierPoseRNN(nn.Module):
    """Same bidirectional-LSTM trunk as models.rnn.RNN, but the output head is
    kinematic-tree structured: every reduced joint decodes a LOCAL 6D rotation
    (as an offset from identity) conditioned on the trunk features AND its
    parent's already-decoded local 6D, then global rotations are composed down
    the chain. Output is the same [B,S,n_reduced*6] reduced-GLOBAL 6D layout as
    the flat head, so MobilePoserNet / IK downstream is unchanged.
    """
    def __init__(self, n_input, n_hidden=256, n_rnn_layer=2, dropout=0.4, head_hidden=128):
        super().__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.rnn = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden, num_layers=n_rnn_layer, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout)
        self.parent_idx = _parent_idx()
        # reduced list is already in topological order (parents before children)
        assert all(p < i for i, p in enumerate(self.parent_idx))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden * 2 + (6 if p >= 0 else 0), head_hidden), nn.ReLU(), nn.Linear(head_hidden, 6))
            for p in self.parent_idx
        ])
        for h in self.heads:
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)
        self.register_buffer('id6d', torch.tensor(IDENTITY_6D))

    def forward(self, x, seq_lengths=None):
        data = self.dropout(relu(self.linear1(x)))
        if seq_lengths is not None:
            data = pack_padded_sequence(data, seq_lengths, batch_first=True, enforce_sorted=False)
        data, _ = self.rnn(data, None)
        if seq_lengths is not None:
            data, _ = pad_packed_sequence(data, batch_first=True)
        B, S, _ = data.shape
        local6d, glob = [], []
        for i, p in enumerate(self.parent_idx):
            inp = data if p < 0 else torch.cat([data, local6d[p]], dim=-1)
            l6 = self.heads[i](inp) + self.id6d
            local6d.append(l6)
            Rl = art.math.r6d_to_rotation_matrix(l6.reshape(-1, 6)).view(B, S, 3, 3)
            glob.append(Rl if p < 0 else glob[p] @ Rl)
        glob6d = torch.stack([art.math.rotation_matrix_to_r6d(g.reshape(-1, 3, 3)).view(B, S, 6) for g in glob], dim=2)
        return glob6d.reshape(B, S, -1), torch.stack(local6d, dim=2)   # [B,S,n*6], [B,S,n,6]


class PoserHier(Poser):
    """Poser with an extra LOCAL-rotation loss and (flat=False) a kinematic-tree
    head. flat=True keeps the original flat RNN head and only adds the local
    loss (control for separating "head" from "loss"). All loss terms are
    masked to the real (un-padded) timesteps: the flat head outputs ~0 on
    padded steps (matching the zero-padded targets) but the tree head always
    emits valid unit-norm rotations, which would otherwise be penalised
    against the zero padding (this inflated val_loss ~12x in the first,
    unmasked run)."""
    def __init__(self, finetune: bool = False, local_weight: float = 1.0, flat: bool = False):
        super().__init__(finetune)
        self.flat = flat
        self.parent_idx = _parent_idx()
        if not flat:
            self.pose = HierPoseRNN(self.C.n_output_joints * 3 + self.C.n_imu)
        self.local_weight = local_weight
        self.save_hyperparameters()

    def _run(self, x, lengths):
        if self.flat:
            return self.pose(x, lengths)[0], None
        return self.pose(x, lengths)

    def forward(self, batch, input_lengths=None):
        return self._run(batch, input_lengths)[0]

    def _local_target(self, pose6d):
        """pose6d [B,S,n_reduced*6] global 6D -> local 6D [B,S,n,6]."""
        B, S, _ = pose6d.shape
        n = joint_set.n_reduced
        R = art.math.r6d_to_rotation_matrix(pose6d.reshape(-1, 6)).view(B, S, n, 3, 3)
        out = []
        for i, p in enumerate(self.parent_idx):
            Rl = R[:, :, i] if p < 0 else R[:, :, p].transpose(-1, -2) @ R[:, :, i]
            out.append(art.math.rotation_matrix_to_r6d(Rl.reshape(-1, 3, 3)).view(B, S, 6))
        return torch.stack(out, dim=2)

    @staticmethod
    def _masked_mse(a, b, mask):
        per_step = ((a - b) ** 2).flatten(2).mean(-1)
        return (per_step * mask).sum() / mask.sum().clamp(min=1)

    def shared_step(self, batch):
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, output_lengths = outputs

        target_pose = outputs['poses']
        B, S, _ = target_pose.shape
        lengths = torch.as_tensor(input_lengths, device=target_pose.device)
        mask = (torch.arange(S, device=target_pose.device)[None] < lengths[:, None]).float()
        target_joints = outputs['joints'].view(B, S, -1)
        noisy_joints = target_joints + torch.randn(target_joints.size()).to(self.C.device) * 0.04

        pose_p, local_p = self._run(torch.cat((noisy_joints, imu_inputs), dim=-1), input_lengths)
        pose_t = target_pose.view(B, S, 24, 6)[:, :, joint_set.reduced].reshape(B, S, -1)

        loss = self._masked_mse(pose_p, pose_t, mask)
        loss += self.t_weight * self.compute_jerk_loss(pose_p)
        if self.local_weight > 0:
            if local_p is None:
                local_p = self._local_target(pose_p)
            loss += self.local_weight * self._masked_mse(local_p, self._local_target(pose_t), mask)
        if self.use_pos_loss:
            full_pose_p = self._reduced_global_to_full(pose_p)
            joints_p = self.bodymodel.forward_kinematics(pose=full_pose_p.view(-1, 216))[1].view(B, S, -1)
            loss += self._masked_mse(joints_p, target_joints, mask)
        return loss
