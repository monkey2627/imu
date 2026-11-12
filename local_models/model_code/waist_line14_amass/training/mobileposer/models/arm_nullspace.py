"""Differentiable elbow reconstruction from the shoulder-wrist null space."""

import torch
import torch.nn.functional as F


def elbow_from_swivel(shoulder, wrist, upper_length, fore_length, swivel, torso_up, eps=1e-6):
    """Return an elbow on the shoulder-wrist sphere-intersection circle.

    Args:
        shoulder, wrist: [..., 3] predicted joint positions.
        upper_length, fore_length: scalar or broadcastable bone lengths.
        swivel: [..., 2] as (sin(angle), cos(angle)).
        torso_up: [..., 3] reference direction used to orient the circle basis.

    The wrist is projected onto the reachable shell when the predicted
    shoulder-wrist distance is outside the two-link reach interval.
    Returns (elbow, projected_wrist, reach_correction).
    """
    if shoulder.shape != wrist.shape or shoulder.shape[-1] != 3:
        raise ValueError("shoulder and wrist must have matching [..., 3] shapes")
    if swivel.shape != shoulder.shape[:-1] + (2,):
        raise ValueError("swivel must have shape [..., 2]")
    if torso_up.shape != shoulder.shape:
        raise ValueError("torso_up must match shoulder shape")

    l1 = torch.as_tensor(upper_length, dtype=shoulder.dtype, device=shoulder.device)
    l2 = torch.as_tensor(fore_length, dtype=shoulder.dtype, device=shoulder.device)
    if torch.any(l1 <= 0) or torch.any(l2 <= 0):
        raise ValueError("bone lengths must be positive")

    delta = wrist - shoulder
    distance = delta.norm(dim=-1, keepdim=True)
    fallback = shoulder.new_tensor([0.0, 0.0, 1.0]).expand_as(delta)
    axis = torch.where(distance > eps, delta / distance.clamp_min(eps), fallback)

    min_reach = (l1 - l2).abs() + eps
    max_reach = (l1 + l2) - eps
    safe_distance = distance.clamp(min=min_reach, max=max_reach)
    projected_wrist = shoulder + axis * safe_distance
    center = shoulder + axis * ((l1.square() - l2.square() + safe_distance.square()) / (2 * safe_distance))
    radius_sq = (l1.square() - ((l1.square() - l2.square() + safe_distance.square()) / (2 * safe_distance)).square()).clamp_min(0)
    radius = radius_sq.sqrt()

    ref = F.normalize(torso_up, dim=-1, eps=eps)
    e1_raw = ref - (ref * axis).sum(-1, keepdim=True) * axis
    fallback_axis = F.one_hot(axis.abs().argmin(-1), num_classes=3).to(dtype=axis.dtype)
    e1_fallback = fallback_axis - (fallback_axis * axis).sum(-1, keepdim=True) * axis
    e1 = F.normalize(torch.where(e1_raw.norm(dim=-1, keepdim=True) > eps, e1_raw, e1_fallback), dim=-1, eps=eps)
    e2 = F.normalize(torch.cross(axis, e1, dim=-1), dim=-1, eps=eps)

    phase_norm = swivel.norm(dim=-1, keepdim=True)
    phase = F.normalize(swivel, dim=-1, eps=eps)
    phase_fallback = torch.zeros_like(swivel)
    phase_fallback[..., 1] = 1
    phase = torch.where(phase_norm > eps, phase, phase_fallback)
    circle_direction = phase[..., 1:2] * e1 + phase[..., 0:1] * e2
    elbow = center + radius * circle_direction
    reach_correction = (projected_wrist - wrist).norm(dim=-1)
    return elbow, projected_wrist, reach_correction
