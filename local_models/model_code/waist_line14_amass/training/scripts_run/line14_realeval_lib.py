"""Shared helpers for stage C (real-motion evaluation of the frozen line-14 downstream networks, with calibration-error injection).

Models = the C21 / C23 end-to-end Joints+Poser pairs (trained from scratch on synthetic AMASS, 60 fps, 125-frame windows, wide-scale + offset population):
  b0     checkpoints/line14/C21/ori-none_geom-none_s{seed}.pt           raw measured orientation, no correction
  i2s    checkpoints/line14/C21/ori-est_geom-none_s{seed}_rot.pt        orientation corrected with the estimated offset
  lvmean checkpoints/line14/C23/ori-est_geom-none_s{seed}_lvmean.pt     i2s + lever-arm compensation features with the population-mean r (no per-person r needed)
Real input = the dataset's own calibrated orientation R = RIM^T RIS RSB (what every baseline receives) and gravity-free world acceleration, [acc/30, R] x5 sensors.
Injection: corrected = R_meas R_est^T with R_est = Exp(delta) R_off  ==>  corrected = R_bone Exp(-delta); so on real data  R_in = R_oracle @ Exp(-delta)  (right multiplication).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mobileposer.articulate as art
from mobileposer.articulate import math as AM
from mobileposer.config import joint_set, paths
from mobileposer.line9.models import GeometryStats, Line9Joints
from mobileposer.utils.model_utils import reduced_pose_to_full
import line14_poser_chain as C
from line14_physfeat import lever_features, prox_vectors

REDUCED = list(joint_set.reduced)
CKPT = {"b0": "checkpoints/line14/C21/ori-none_geom-none_s{seed}.pt",
        "i2s": "checkpoints/line14/C21/ori-est_geom-none_s{seed}_rot.pt",
        "lvmean": "checkpoints/line14/C23/ori-est_geom-none_s{seed}_lvmean.pt"}
MEAN_R_FILE = "data_local_mean_r.pt"


def mean_r():
    """Population-mean sensor->proximal-joint vectors [4,3] over the 342 TRAINING identities (as in line14_e2e_link.py --lever mean)."""
    if Path(MEAN_R_FILE).is_file():
        return torch.load(MEAN_R_FILE)
    from mobileposer.line9.data import load_identity_records
    from mobileposer.line14.geom import GeomProvider
    rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("train",))["train"]
    g = GeomProvider("r2", 3.0, None, None)
    betas = {}
    for r in rec:
        betas.setdefault(r.identity, torch.tensor(r.beta, dtype=torch.float32).reshape(-1)[:10])
    m = torch.stack([prox_vectors(g.base(b, i)) for i, b in betas.items()]).mean(0)
    torch.save(m, MEAN_R_FILE)
    return m


def load_model(arm: str, seed: int, device):
    extra = 36 if arm == "lvmean" else 0
    jnet = Line9Joints("b0", geom_dim=1, geometry_stats=GeometryStats(torch.zeros(1), torch.ones(1)))
    if extra:
        jnet.linear1 = nn.Linear(60 + extra, 256)
    poser = C.ChainPoser(132 + extra)
    st = torch.load(CKPT[arm].format(seed=seed), map_location="cpu", weights_only=False)
    jnet.load_state_dict(st["jnet"]); poser.load_state_dict(st["poser"])
    return jnet.to(device).eval(), poser.to(device).eval(), extra


@torch.no_grad()
def infer(models, imu: torch.Tensor, r=None, W: int = 125, stride: int = 63, bs: int = 96):
    """imu [T,60] (acc/30 | orientation, already corrected) -> reduced global 6D [T,96].  Overlapping windows, central part kept (WinPred rule)."""
    jnet, poser, extra = models
    dev = next(jnet.parameters()).device
    T = len(imu)
    starts = list(range(0, T - W, stride)) + [T - W]
    m = (W - stride) // 2
    out = torch.zeros(T, 96)
    for i in range(0, len(starts), bs):
        ss = starts[i:i + bs]
        x = torch.stack([imu[s:s + W] for s in ss]).to(dev)
        L = [W] * len(ss)
        if extra:
            x = torch.cat((x, lever_features(x, r.to(dev).expand(len(ss), 4, 3), L)), dim=-1)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            j = jnet(x, L)
            p = poser(torch.cat((j.to(x.dtype), x), -1), L)
        p = p.float().cpu()
        for k, s in enumerate(ss):
            idx = i + k
            a = 0 if idx == 0 else s + m
            b = T if idx == len(starts) - 1 else s + W - m
            out[a:b] = p[k, a - s:b - s]
    return out


def to_local(body, p96: torch.Tensor, device):
    """[T,96] reduced global 6D -> local rotation matrices [T,24,3,3] with the ignored joints set to identity (as in line14_poser_chain.evaluate)."""
    T = len(p96)
    Rr = AM.r6d_to_rotation_matrix(p96.reshape(-1, 6)).reshape(T, 16, 3, 3).to(device)
    full = torch.eye(3, device=device).repeat(T, 24, 1, 1)
    full[:, REDUCED] = Rr
    local = body.inverse_kinematics_R(full)
    for ig in joint_set.ignored:
        local[:, ig] = torch.eye(3, device=device)
    return local


def inject(R: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """R [T,5,3,3] oracle-calibrated orientation, delta [5,3] radians (left tangent error of the offset estimate) -> R @ Exp(-delta), constant over the sequence."""
    E = AM.axis_angle_to_rotation_matrix((-delta).reshape(-1, 3).float()).reshape(5, 3, 3)
    return R @ E[None]


def log_so3(R):
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1) / 2
    n = v.norm(dim=-1, keepdim=True); c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).unsqueeze(-1)
    return torch.where(n > 1e-12, v / n.clamp_min(1e-12) * torch.atan2(n, c), v)
