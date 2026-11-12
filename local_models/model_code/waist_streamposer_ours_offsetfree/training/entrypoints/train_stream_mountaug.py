#!/usr/bin/env python3
"""train_ours_balanced_stream.py with sensor-error augmentation matched to the live residuals.

The stock --mount-deg draws one body-frame offset sigma for all five sensors.
Live, the errors left after calibration are larger on the forearms (twist is
never calibrated, heading only once per set) and include world-frame heading
errors, which also rotate SlimeVR's world acceleration. Training batches
(Ours and AMASS replay) get, per sequence and sensor:
  ori' = Yaw(h) @ ori @ Mount(m),  acc' = Yaw(h) @ acc
with h ~ N(0, HEADING_SIGMA) and m a random-axis rotation of |N(0, MOUNT_SIGMA)|.
Evaluation is untouched (it calls batch_inputs with mount_deg = 0).
All other arguments are passed through; --mount-deg only needs to be > 0.
"""
import json
import runpy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

MOUNT_SIGMA_DEG = [30.0, 30.0, 15.0, 15.0, 10.0]     # lw rw ls rs waist (body frame)
HEADING_SIGMA_DEG = [25.0, 25.0, 10.0, 10.0, 0.0]    # world yaw; the waist is the heading reference

workspace = sys.argv[sys.argv.index('--workspace') + 1]
sys.path.insert(0, workspace)
import mobileposer.stream.data as stream_data            # noqa: E402
from mobileposer.articulate import math as art_math      # noqa: E402
from mobileposer.stream.features import imu_features     # noqa: E402


def _mount(n, device):
    sigma = torch.tensor(MOUNT_SIGMA_DEG, device=device).view(1, 5, 1) * torch.pi / 180
    axis = F.normalize(torch.randn(n, 5, 3, device=device), dim=-1)
    angle = torch.randn(n, 5, 1, device=device).abs() * sigma
    return art_math.axis_angle_to_rotation_matrix((axis * angle).view(-1, 3)).view(n, 5, 3, 3)


def _heading(n, device):
    angle = torch.randn(n, 5, device=device) * torch.tensor(HEADING_SIGMA_DEG, device=device) * torch.pi / 180
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, zero, s, zero, one, zero, -s, zero, c), dim=-1).view(n, 5, 3, 3)


def batch_inputs(acc, ori, gyro, mount_deg=0.0, frame='rel', local=False):
    if mount_deg > 0:
        n = len(ori)
        yaw = _heading(n, ori.device).unsqueeze(1).to(ori.dtype)
        ori = yaw @ ori @ _mount(n, ori.device).unsqueeze(1).to(ori.dtype)
        acc = (yaw @ acc.unsqueeze(-1)).squeeze(-1)
    return imu_features(acc, ori, gyro, frame, local), ori


stream_data.batch_inputs = batch_inputs
output = Path(sys.argv[sys.argv.index('--output') + 1])
script = Path(__file__).with_name('train_ours_balanced_stream.py')
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name='__main__')
(output / 'augmentation.json').write_text(json.dumps({
    'mount_sigma_deg': dict(zip(['lw', 'rw', 'ls', 'rs', 'waist'], MOUNT_SIGMA_DEG)),
    'heading_sigma_deg': dict(zip(['lw', 'rw', 'ls', 'rs', 'waist'], HEADING_SIGMA_DEG)),
    'applies_to': 'Ours and AMASS replay training batches; evaluation unaugmented'}, indent=2))
