import os
os.environ["MOBILEPOSER_FPS"] = "60"
import torch
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.line9.calib import CONFIG, identity_offsets, synth_sequence
from mobileposer.surface_imu import load_attachments, surface_kinematics
from mobileposer.line14.protocol import make_dynamic_protocol, randomized_pose_p
t = torch.load("data_local/line14_cal_dyn13_dev10/train.pt", weights_only=False)
key, beta = t["key"][15], t["beta"][15]
proto = make_dynamic_protocol()
R, phase, psi = randomized_pose_p(proto, key, 10.0)
print("R finite", torch.isfinite(R).all().item(), "frames", len(R))
M = ParametricModel(paths.smpl_file); C = load_attachments(CONFIG, M)
from mobileposer.line9.calib import LABELS
rot, _, pts = surface_kinematics(M, R, beta, torch.zeros(len(R), 3), LABELS, C, 128)
print("rot finite", torch.isfinite(rot).all().item(), "pts finite", torch.isfinite(pts).all().item())
bad = (~torch.isfinite(rot).reshape(len(rot), -1).all(1)).nonzero().flatten()
print("bad rot frames", bad[:20].tolist())
if len(bad):
    f = int(bad[0]); print("pose around bad frame max |aa|:", proto.pose_aa[min(f, len(proto.pose_aa)-1)].abs().max().item())
    Rf = R[f]; print("R[f] finite", torch.isfinite(Rf).all().item(), "det range", torch.linalg.det(Rf).min().item(), torch.linalg.det(Rf).max().item())
