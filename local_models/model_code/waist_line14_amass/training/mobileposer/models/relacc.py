import os
import numpy as np
import torch

N_EXTRA = 24


def use_relacc():
    return os.environ.get('MOBILEPOSER_RELACC') == '1'


def rel_acc_features(imu):
    """Explicit relative accelerations of the four limb sensors w.r.t. the waist sensor (slot 4).
    imu [..., 60] = 5 x acc (3, already /acc_scale) + 5 x orientation (3x3).
    Returns [..., 24]: (a_k - a_waist) in the waist frame (12) and in the sensor's own frame (12).
    Gravity cancels in the difference; the frame choice makes the cue invariant to heading."""
    acc = imu[..., :15].reshape(*imu.shape[:-1], 5, 3)
    ori = imu[..., 15:60].reshape(*imu.shape[:-1], 5, 3, 3)
    rel = acc[..., :4, :] - acc[..., 4:5, :]                                          # [...,4,3]
    in_waist = (ori[..., 4:5, :, :].transpose(-1, -2) @ rel.unsqueeze(-1)).squeeze(-1)
    in_own = (ori[..., :4, :, :].transpose(-1, -2) @ rel.unsqueeze(-1)).squeeze(-1)
    return 3.0 * torch.cat([in_waist.flatten(-2), in_own.flatten(-2)], dim=-1)


N_LEVER = 36


def use_lever():
    return os.environ.get('MOBILEPOSER_LEVER') in ('1', 'arms')


def n_ctx():
    return 37 if os.environ.get('MOBILEPOSER_CTX') == '1' else 0


def _rotvec(R):
    """[...,3,3] -> [...,3] axis-angle (log map)"""
    import mobileposer.articulate as art
    sh = R.shape[:-2]
    return art.math.rotation_matrix_to_axis_angle(R.reshape(-1, 3, 3)).reshape(*sh, 3)


def lever_features(imu, dt=1.0 / 30, acc_scale=30.0):
    """Lever-arm compensated accelerations (the model information from the SMPL template: where the sensor sits).
    imu [B,S,97] = acc(5x3, /acc_scale) | ori(5x3x3) | context(37; first 12 = lever arms r of the 4 limb sensors, bone frame).
    A limb sensor measures a_s = a_J + alpha x r_w + w x (w x r_w) (J = proximal joint of the bone; r_w = R r).  w, alpha come
    from the (accurate) sensed orientations by finite differences, r from the template, so a_J - a_waist, the translational
    acceleration of the UNSENSED joint (elbow / knee) relative to the waist, can be formed.
    Returns [B,S,36]: (a_J - a_waist) in the waist frame (12) and in the sensor frame (12), rotational term in the waist frame (12); /acc_scale*3."""
    B, S = imu.shape[:2]
    acc = imu[..., :15].reshape(B, S, 5, 3) * acc_scale
    R = imu[..., 15:60].reshape(B, S, 5, 3, 3)
    r = imu[..., 60:72].reshape(B, S, 4, 3)
    Rl = R[:, :, :4]
    w_loc = torch.zeros(B, S, 4, 3, device=imu.device, dtype=imu.dtype)
    if S >= 3:
        d = Rl[:, :-2].transpose(-1, -2) @ Rl[:, 2:]
        w_loc[:, 1:-1] = _rotvec(d) / (2 * dt)
        w_loc[:, 0], w_loc[:, -1] = w_loc[:, 1], w_loc[:, -2]
    w = (Rl @ w_loc.unsqueeze(-1)).squeeze(-1)                                                  # world angular velocity
    alpha = torch.zeros_like(w)
    if S >= 3:
        alpha[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2 * dt)
        alpha[:, 0], alpha[:, -1] = alpha[:, 1], alpha[:, -2]
    rw = (Rl @ r.unsqueeze(-1)).squeeze(-1)
    term = torch.cross(alpha, rw, dim=-1) + torch.cross(w, torch.cross(w, rw, dim=-1), dim=-1)
    rel = acc[:, :, :4] - term - acc[:, :, 4:5]                                                # joint accel. relative to the waist (world)
    Rw_T = R[:, :, 4:5].transpose(-1, -2)
    f1 = (Rw_T @ rel.unsqueeze(-1)).squeeze(-1)
    f2 = (Rl.transpose(-1, -2) @ rel.unsqueeze(-1)).squeeze(-1)
    f3 = (Rw_T @ term.unsqueeze(-1)).squeeze(-1)
    if os.environ.get("MOBILEPOSER_LEVER") == "arms":                                           # shank compensation hurts on real data (sim-to-real): arms only
        keep = torch.tensor([1.0, 1.0, 0.0, 0.0], device=imu.device, dtype=imu.dtype).view(1, 1, 4, 1)
        f1, f2, f3 = f1 * keep, f2 * keep, f3 * keep
    return torch.cat([f1.flatten(-2), f2.flatten(-2), f3.flatten(-2)], dim=-1) * (3.0 / acc_scale)


def augment(batch, n_imu, relacc, lever):
    """append the explicit physical features computed from the trailing IMU(+context) columns"""
    imu = batch[..., -n_imu:]
    extra = []
    if relacc:
        extra.append(rel_acc_features(imu))
    if lever:
        extra.append(lever_features(imu))
    if use_integ():
        extra.append(integrator_features(imu))
    return torch.cat([batch] + extra, dim=-1) if extra else batch


N_INTEG = 48


def use_integ():
    return os.environ.get('MOBILEPOSER_INTEG') == '1'


def integrator_features(imu, ns=None, taus=(0.3, 1.0)):
    """Leaky double integration of the RELATIVE acceleration of the four limb sensors w.r.t. the pelvis sensor (world frame, gravity-free).
    The relative acceleration of a forearm / shank is reliable (R2 0.9 / 0.8 against the reference at 60 fps), but a recurrent network has to
    integrate it twice to see what it implies: how far and how fast the wrist / ankle moved relative to the pelvis over the last 0.3 s / 1 s.
    Returns [B,S,48]: (velocity, displacement) x 2 time constants x 4 sensors x 3, expressed in the pelvis HEADING frame."""
    from mobileposer.frames import frame_rot
    ns = ns or int(os.environ.get('MOBILEPOSER_NSENS', 5))
    B, S = imu.shape[:2]
    fps = int(os.environ.get('MOBILEPOSER_FPS', 30))
    dt = 1.0 / fps
    acc = imu[..., :3 * ns].reshape(B, S, ns, 3) * 30.0
    pel = ns - 1 if ns == 5 else ns - 1
    rel = acc[:, :, :4] - acc[:, :, pel:pel + 1]                                   # [B,S,4,3]
    feats = []
    for tau in taus:
        lam = float(np.exp(-dt / tau))
        v = torch.zeros(B, 4, 3, device=imu.device, dtype=imu.dtype)
        p = torch.zeros_like(v)
        V, P = [], []
        for t in range(S):
            v = lam * v + rel[:, t] * dt
            p = lam * p + v * dt
            V.append(v); P.append(p)
        feats += [torch.stack(V, 1), 5.0 * torch.stack(P, 1)]
    F = torch.cat(feats, dim=2)                                                     # [B,S,16,3]
    Rp = imu[..., 3 * ns:12 * ns].reshape(B, S, ns, 3, 3)[:, :, pel]
    Rf = frame_rot(Rp.reshape(-1, 3, 3), 'yaw').reshape(B, S, 3, 3)
    F = (Rf.transpose(-1, -2).unsqueeze(2) @ F.unsqueeze(-1)).squeeze(-1)
    return F.flatten(2)
