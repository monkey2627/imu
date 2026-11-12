"""PivotPrint used as PHYSICS (line 14, C23).

A limb sensor measures  a_s = a_J + alpha x r_w + w x (w x r_w)   (J = the proximal joint of the sensed bone: elbow for the forearm sensors, knee for the shank
sensors; r_w = R_bone r with r = the sensor->joint vector of PivotPrint; w, alpha from finite differences of the (offset-corrected) measured orientations).
With r known, a_J - a_pelvis (translational acceleration of the UNSENSED elbow / knee relative to the pelvis) can be formed; it is a direct cue for the motion of
the unsensed upper arm / thigh, and its scale follows the body size through r.  This is how the calibrated vectors enter the computation with their direction and
length, instead of only rescaling a FiLM-conditioned network.

lever_features(imu, r, lengths) -> [B,T,36]: (a_J - a_pelvis) in the pelvis frame (12) | in the sensor frame (12) | rotational term in the pelvis frame (12),  * 3/30
"""
from __future__ import annotations
import torch
from mobileposer.articulate import math as AM


def _rotvec(R):
    sh = R.shape[:-2]
    return AM.rotation_matrix_to_axis_angle(R.reshape(-1, 3, 3)).reshape(*sh, 3)


def prox_vectors(r2_flat):
    """24-d R2 (8 sensor->joint vectors: lw elbow/shoulder, rw elbow/shoulder, ls knee/hip, rs knee/hip) -> [4,3] sensor->proximal-joint (elbow, elbow, knee, knee)"""
    return r2_flat.reshape(8, 3)[[0, 2, 4, 6]]


@torch.no_grad()
def lever_features(imu, r, lengths, fps=60, acc_scale=30.0, keep=(1, 1, 1, 1)):
    """imu [B,T,60] (acc/30 | ori, after the orientation correction), r [B,4,3] metres, lengths: valid frames per row."""
    with torch.autocast(device_type="cuda", enabled=False):
        imu = imu.float(); r = r.float().to(imu.device)
        B, T = imu.shape[:2]; dt = 1.0 / fps
        valid = torch.arange(T, device=imu.device)[None] < torch.as_tensor(lengths, device=imu.device)[:, None]            # [B,T]
        eye = torch.eye(3, device=imu.device)
        R = imu[..., 15:60].reshape(B, T, 5, 3, 3)
        R = torch.where(valid[:, :, None, None, None], R, eye)                                                              # padded steps -> identity (no NaN)
        acc = imu[..., :15].reshape(B, T, 5, 3) * acc_scale
        Rl = R[:, :, :4]
        w_loc = torch.zeros(B, T, 4, 3, device=imu.device)
        if T >= 3:
            d = Rl[:, :-2].transpose(-1, -2) @ Rl[:, 2:]
            w_loc[:, 1:-1] = _rotvec(d) / (2 * dt); w_loc[:, 0] = w_loc[:, 1]; w_loc[:, -1] = w_loc[:, -2]
        w = (Rl @ w_loc.unsqueeze(-1)).squeeze(-1)
        alpha = torch.zeros_like(w)
        if T >= 3:
            alpha[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2 * dt); alpha[:, 0] = alpha[:, 1]; alpha[:, -1] = alpha[:, -2]
        rw = (Rl @ r[:, None, :, :, None]).squeeze(-1)                                                                      # [B,T,4,3]
        term = torch.cross(alpha, rw, dim=-1) + torch.cross(w, torch.cross(w, rw, dim=-1), dim=-1)
        rel = acc[:, :, :4] - term - acc[:, :, 4:5]
        Rp_T = R[:, :, 4:5].transpose(-1, -2)
        f1 = (Rp_T @ rel.unsqueeze(-1)).squeeze(-1)
        f2 = (Rl.transpose(-1, -2) @ rel.unsqueeze(-1)).squeeze(-1)
        f3 = (Rp_T @ term.unsqueeze(-1)).squeeze(-1)
        kp = torch.tensor(keep, dtype=torch.float32, device=imu.device).view(1, 1, 4, 1)          # sensors whose compensation is used (lw, rw, ls, rs)
        f1, f2, f3 = f1 * kp, f2 * kp, f3 * kp
        out = torch.cat([f1.flatten(-2), f2.flatten(-2), f3.flatten(-2)], dim=-1) * (3.0 / acc_scale)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0) * valid[..., None]
    return out
