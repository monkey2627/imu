"""PosAccPoser: StreamPoser ('mp2', frame 'both') with an explicit, initialisable sensor-motion stage (2026-11-06).

Why: StreamPoser ignores the acceleration even on clean synthetic input (bird-dog knee error 91 deg with or
without it), while PNP's acceleration path halves the error.  What PNP has and StreamPoser lacks:
  - a stage that tracks the limb sensors' POSITIONS (here also their VELOCITIES, a short integration that
    forces the acceleration to be used) relative to the waist sensor, supervised directly;
  - a recurrent state initialised from the known starting pose (PNP's rnn_initialize), so the integration
    has an origin; with the pose unknown the init network gets zeros and a 'known' flag of 0.
Stages (causal LSTMs, state kept across calls, so it streams like StreamPoser):
  P: IMU features -> sensor positions (12) + velocities (12), pelvis frame, relative to the waist sensor
  J: features + P -> 23 root-relative joint positions (69)        (as StreamPoser stage 1)
  R: features + J -> reduced root-relative global rotations (90)  (as StreamPoser stage 2)
Outputs and compose() match StreamPoser (anchor 'none'), so existing evaluation/adapters can be reused.
Copy of server_training_20261030/posacc_model.py (the training code); imports the existing feature code read-only.
"""
import torch
import torch.nn as nn

from mobileposer.stream.features import n_features, compose, N_JOINT_OUT, N_ROT_OUT

N_SENS = 4                       # limb sensors lw, rw, ls, rs (waist is the reference)
N_PV = 2 * 3 * N_SENS            # positions + velocities
# sensor points along their bones (no_head_surface_imu.json bone_fraction): (parent joint, child joint, fraction)
SENSOR_POINTS = ((18, 20, 0.75), (19, 21, 0.75), (4, 7, 0.88), (5, 8, 0.88))
WAIST_JOINT = 3


def sensor_targets(jrel, fps=60):
    """jrel [B,T,69] (joints 1..23, pelvis frame) -> positions and velocities [B,T,24] of the limb sensor
    points relative to the waist sensor point (spine1), pelvis frame."""
    J = jrel.view(*jrel.shape[:2], 23, 3)
    at = lambda j: J[:, :, j - 1]
    waist = at(WAIST_JOINT)
    pos = torch.stack([at(a) + f * (at(b) - at(a)) - waist for a, b, f in SENSOR_POINTS], 2).flatten(2)
    vel = torch.zeros_like(pos)
    vel[:, 1:-1] = (pos[:, 2:] - pos[:, :-2]) * fps / 2
    vel[:, 0], vel[:, -1] = vel[:, 1], vel[:, -2]
    return torch.cat([pos, vel], -1)


class PosAccPoser(nn.Module):
    def __init__(self, gyro=True, hidden=512, layers=3, dropout=0.2, frame='both'):
        super().__init__()
        nf = n_features(gyro, frame, False)
        self.gyro, self.frame, self.local, self.anchor, self.arch = gyro, frame, False, 'none', 'posacc'
        self.hidden, self.layers = hidden, layers
        self.init_net = nn.Sequential(nn.Linear(3 * N_SENS + 1, 256), nn.ReLU(), nn.Linear(256, 2 * layers * hidden))
        self.inp0 = nn.Sequential(nn.Linear(nf, hidden), nn.ReLU())
        self.rnn0 = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout)
        self.out0 = nn.Linear(hidden, N_PV)
        self.inp1 = nn.Sequential(nn.Linear(nf + N_PV, hidden), nn.ReLU())
        self.rnn1 = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout)
        self.out1 = nn.Linear(hidden, N_JOINT_OUT)
        self.inp2 = nn.Sequential(nn.Linear(nf + N_JOINT_OUT, hidden), nn.ReLU())
        self.rnn2 = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout)
        self.out2 = nn.Linear(hidden, N_ROT_OUT)
        self.aux = {}

    def compose(self, out, ori_rel):
        return compose(out, ori_rel, 'none')

    def initial_state(self, batch, device, init_pos=None, known=None):
        """LSTM state of stage P from the starting sensor positions [B,12]; known [B] (bool) marks which
        sequences have a known start (default: all if init_pos is given, none otherwise)."""
        if init_pos is None:
            z = torch.zeros(batch, 3 * N_SENS + 1, device=device)
        else:
            k = torch.ones(batch, device=device) if known is None else known.float()
            z = torch.cat([init_pos * k[:, None], k[:, None]], -1)
        hc = self.init_net(z).view(batch, 2, self.layers, self.hidden).permute(1, 2, 0, 3).contiguous()
        return hc[0], hc[1]

    def forward(self, x, state=None, init_pos=None, known=None):
        """x [B,T,F] -> joints [B,T,69], rotations [B,T,90], new state; self.aux['pv'] = stage-P output."""
        if state is None:
            state = (self.initial_state(x.shape[0], x.device, init_pos, known), None, None)
        s0, s1, s2 = state
        h0, s0 = self.rnn0(self.inp0(x), s0)
        pv = self.out0(h0)
        h1, s1 = self.rnn1(self.inp1(torch.cat([x, pv], -1)), s1)
        joints = self.out1(h1)
        h2, s2 = self.rnn2(self.inp2(torch.cat([x, joints], -1)), s2)
        self.aux = {'pv': pv}
        return joints, self.out2(h2), (s0, s1, s2)
