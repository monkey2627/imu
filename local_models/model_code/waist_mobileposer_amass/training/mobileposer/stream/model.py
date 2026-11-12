"""Causal unidirectional LSTM cascades; every stage keeps its recurrent state across calls, so a sequence can be streamed.

arch 'mp2' (MobilePoser-style, default): IMU features -> root-relative joint positions -> root-relative reduced global rotations.
arch 'pnp3' (PNP/PIP-style): IMU -> leaf-joint positions (wrists, ankles, head) -> all joint positions -> rotations, where
the last stage sees only the pelvis-relative sensor orientations (+ up vector) and the joint positions, not accelerations."""
import torch
import torch.nn as nn

from mobileposer.stream.features import n_features, compose, N_JOINT_OUT, N_ROT_OUT, N_EXTRA

LEAF = [20, 21, 7, 8, 15]                  # wrists, ankles, head (SMPL joints); index j-1 in the 23 root-relative joints
N_LEAF = 3 * len(LEAF)
N_ORI_REL = 36 + 3                         # leading block of the 'rel' features: limb orientations + up vector


def _stage(n_in, n_out, hidden, layers, dropout):
    return nn.ModuleDict({'inp': nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU()),
                          'rnn': nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout),
                          'out': nn.Linear(hidden, n_out)})


def _run(stage, x, s):
    h, s = stage['rnn'](stage['inp'](x), s)
    return stage['out'](h), s


class StreamPoser(nn.Module):
    def __init__(self, gyro=True, hidden=512, layers=3, dropout=0.2, anchor='none', frame='rel', arch='mp2', local=False):
        super().__init__()
        nf = n_features(gyro, frame, local)
        self.gyro = gyro
        self.frame = frame            # input representation, see features.FRAMES
        self.local = local            # + limb angular velocity in the sensors' own frames
        self.anchor = anchor          # see features.ANCHORS; r6d outputs of anchored joints are relative to their sensor
        self.arch = arch
        n_rot = N_ROT_OUT + N_EXTRA.get(anchor, 0)
        if arch == 'mp2':
            self.inp1 = nn.Sequential(nn.Linear(nf, hidden), nn.ReLU())
            self.rnn1 = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout)
            self.out1 = nn.Linear(hidden, N_JOINT_OUT)
            self.inp2 = nn.Sequential(nn.Linear(nf + N_JOINT_OUT, hidden), nn.ReLU())
            self.rnn2 = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout)
            self.out2 = nn.Linear(hidden, n_rot)
        elif arch == 'pnp3':
            assert frame in ('rel', 'both'), "pnp3 reads the pelvis-relative orientation block of the features"
            self.leaf = _stage(nf, N_LEAF, hidden, layers, dropout)
            self.jpos = _stage(nf + N_LEAF, N_JOINT_OUT, hidden, layers, dropout)
            self.ik = _stage(N_ORI_REL + N_JOINT_OUT, n_rot, hidden, layers, dropout)
        else:
            raise ValueError(arch)
        self.aux = {}                 # intermediate outputs of the last forward (pnp3: 'leaf') for auxiliary losses
        # 'armtw' constants (set from AMASS before training, saved with the weights, never trained)
        if anchor == 'armtw':
            self.register_buffer('arm_axes', torch.zeros(2, 3))
            self.register_buffer('arm_tau', torch.zeros(2))
            self.register_buffer('arm_bound', torch.zeros(()))
        else:
            self.arm_axes = self.arm_tau = self.arm_bound = None

    def compose(self, out, ori_rel):
        return compose(out, ori_rel, self.anchor, self.arm_axes, self.arm_tau, self.arm_bound)

    def forward(self, x, state=None):
        """x [B,T,F] -> joints [B,T,69], raw rotation output [B,T,90(+extra)] (pass through self.compose), new state."""
        if self.arch == 'pnp3':
            s1, s2, s3 = state if state is not None else (None, None, None)
            leaf, s1 = _run(self.leaf, x, s1)
            joints, s2 = _run(self.jpos, torch.cat([x, leaf], -1), s2)
            rot, s3 = _run(self.ik, torch.cat([x[..., :N_ORI_REL], joints], -1), s3)
            self.aux = {'leaf': leaf}
            return joints, rot, (s1, s2, s3)
        s1, s2 = state if state is not None else (None, None)
        h1, s1 = self.rnn1(self.inp1(x), s1)
        joints = self.out1(h1)
        h2, s2 = self.rnn2(self.inp2(torch.cat([x, joints], -1)), s2)
        return joints, self.out2(h2), (s1, s2)
