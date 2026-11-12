"""R1 geometric representation (line 14): vectors from the pelvis IMU to the other four IMUs,
expressed in the SMPL pelvis/global axes with the body in the SMPL zero pose (T pose).

R1(beta) = [p_lw - p_pel, p_rw - p_pel, p_ls - p_pel, p_rs - p_pel]  (12 values, metres)
where p_* are the PNP surface-attachment points on the zero-pose mesh of shape ``beta``.
Mount offsets are NOT varied here: the attachment points are the fixed PNP vertices, so
R1 is a deterministic function of beta in this first measurement.
"""
import json
from pathlib import Path
import torch
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.surface_imu import attachment_points
from mobileposer.line9.data import _beta_key

LABELS = ["lw", "rw", "ls", "rs", "pelvis"]
CONFIG = Path(__file__).resolve().parents[1] / "configs/pnp_surface_imu.json"


class R1Table:
    def __init__(self, config_path=CONFIG):
        self.body = ParametricModel(paths.smpl_file)
        self.config = json.loads(Path(config_path).read_text())
        self.cache = {}

    def get(self, beta):
        beta = torch.as_tensor(beta, dtype=torch.float32).reshape(-1)[:10]
        key = _beta_key(beta)
        if key not in self.cache:
            with torch.no_grad():
                _, verts = self.body.get_zero_pose_joint_and_vertex(beta)
                pts = attachment_points(verts.reshape(1, -1, 3), self.config, LABELS)[0]
            self.cache[key] = (pts[:4] - pts[4]).reshape(12).float()
        return self.cache[key]

    def points_joints(self, beta):
        """Zero-pose sensor points [5,3] (labels order lw,rw,ls,rs,pelvis) and joints [24,3] of shape ``beta``."""
        beta = torch.as_tensor(beta, dtype=torch.float32).reshape(-1)[:10]
        key = "pj|" + _beta_key(beta)
        if key not in self.cache:
            with torch.no_grad():
                joints, verts = self.body.get_zero_pose_joint_and_vertex(beta)
                pts = attachment_points(verts.reshape(1, -1, 3), self.config, LABELS)[0]
            self.cache[key] = (pts.float(), joints.reshape(24, 3).float())
        return self.cache[key]

    def batch(self, betas):
        return torch.stack([self.get(b) for b in betas.detach().cpu()])
