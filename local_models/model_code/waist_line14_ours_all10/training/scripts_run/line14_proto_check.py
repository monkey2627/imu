"""Sanity checks for the dynamic calibration protocol."""
import torch
from mobileposer.line14.protocol import make_dynamic_protocol
from mobileposer.line9.fixed_protocol import validate_protocol, make_protocol_pose

p = make_dynamic_protocol()
print("frames", p.frames, "seconds", p.seconds, "phases", sorted(set(p.phase_id.tolist())))
print(validate_protocol(p))


def peak(pr):
    d = (pr.pose_aa[1:] - pr.pose_aa[:-1]).norm(dim=-1).max(-1).values * 60
    return float(d.max()), float(d.mean())


print("dynamic peak/mean joint speed (rad/s):", peak(p))
print("slow    peak/mean joint speed (rad/s):", peak(make_protocol_pose()))
