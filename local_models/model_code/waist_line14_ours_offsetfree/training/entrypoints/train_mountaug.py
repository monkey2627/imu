#!/usr/bin/env python3
"""Run train_ours_balanced_{mobileposer,line14}.py with the live-matched sensor-error augmentation.

Same augmentation as train_stream_mountaug.py: per crop and sensor,
  ori' = Yaw(h) @ ori @ Mount(m),  acc' = Yaw(h) @ acc
with h ~ N(0, HEADING_SIGMA) and m a random-axis rotation of |N(0, MOUNT_SIGMA)|.
Both scripts build the 60-D input (15 acc/30, then 45 orientation values) inline
and pass it to a module-level forward()/predict(); that function is wrapped so
the input is perturbed only while gradients are enabled (training), never in
their @torch.no_grad() evaluation.

Usage: train_mountaug.py {mobileposer|line14} <all arguments of the original script>
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

MOUNT_SIGMA_DEG = [30.0, 30.0, 15.0, 15.0, 10.0]     # lw rw ls rs waist (body frame)
HEADING_SIGMA_DEG = [25.0, 25.0, 10.0, 10.0, 0.0]    # world yaw; the waist is the heading reference
TARGETS = {"mobileposer": ("train_ours_balanced_mobileposer.py", "forward"),
           "line14": ("train_ours_balanced_line14.py", "predict")}


def _axis_angle(vectors):
    angle = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = vectors / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).view(*vectors.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=vectors.dtype, device=vectors.device).expand_as(k)
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return eye + s * k + (1 - c) * (k @ k)


def perturb(imu):
    """imu [B, T, 60] -> same layout with one constant mount + heading error per crop and sensor."""
    b, device, dtype = imu.shape[0], imu.device, imu.dtype
    acc = imu[..., :15].reshape(b, -1, 5, 3)
    ori = imu[..., 15:].reshape(b, -1, 5, 3, 3)
    sigma = torch.tensor(MOUNT_SIGMA_DEG, device=device, dtype=dtype).view(1, 5, 1) * torch.pi / 180
    axis = F.normalize(torch.randn(b, 5, 3, device=device, dtype=dtype), dim=-1)
    mount = _axis_angle(axis * torch.randn(b, 5, 1, device=device, dtype=dtype).abs() * sigma)
    h = torch.randn(b, 5, device=device, dtype=dtype) * torch.tensor(HEADING_SIGMA_DEG, device=device, dtype=dtype) * torch.pi / 180
    c, s, zero, one = torch.cos(h), torch.sin(h), torch.zeros_like(h), torch.ones_like(h)
    yaw = torch.stack((c, zero, s, zero, one, zero, -s, zero, c), dim=-1).view(b, 5, 3, 3)
    ori = yaw[:, None] @ ori @ mount[:, None]
    acc = (yaw[:, None] @ acc.unsqueeze(-1)).squeeze(-1)
    return torch.cat((acc.flatten(2), ori.flatten(2)), dim=-1)


def main():
    kind = sys.argv.pop(1)
    script_name, function_name = TARGETS[kind]
    script = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location("train_target", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = getattr(module, function_name)

    def augmented(model_or_jnet, *rest):
        imu = rest[-1]
        if torch.is_grad_enabled():
            imu = perturb(imu)
        return original(model_or_jnet, *rest[:-1], imu)

    setattr(module, function_name, augmented)
    sys.argv[0] = str(script)
    module.main()
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    (output / 'augmentation.json').write_text(json.dumps({
        'mount_sigma_deg': dict(zip(['lw', 'rw', 'ls', 'rs', 'waist'], MOUNT_SIGMA_DEG)),
        'heading_sigma_deg': dict(zip(['lw', 'rw', 'ls', 'rs', 'waist'], HEADING_SIGMA_DEG)),
        'applies_to': 'Ours and AMASS replay training crops; evaluation unaugmented'}, indent=2))


if __name__ == '__main__':
    main()
