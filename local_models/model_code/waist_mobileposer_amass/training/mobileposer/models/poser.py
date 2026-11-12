import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np

from mobileposer.config import *
from mobileposer.utils.model_utils import reduced_pose_to_full
import mobileposer.articulate as art
from mobileposer.models.rnn import RNN
from mobileposer.models.seq import make_seq_model, arch
from mobileposer.models.relacc import N_EXTRA, N_LEVER, N_INTEG, use_relacc, use_lever, use_integ, augment
from mobileposer.models.imu_renderer import render_imu
from mobileposer.surface_imu import load_attachments
from pathlib import Path


class Poser(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations).
    """
    def __init__(self, finetune: bool=False):
        super().__init__()
        
        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = finetune_hypers if finetune else train_hypers

        # body model
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R

        # model definitions
        self.relacc, self.lever = use_relacc(), use_lever()
        self.shortcut = os.environ.get("MOBILEPOSER_SHORTCUT") == "1"
        self.anchor = set(filter(None, os.environ.get("MOBILEPOSER_ANCHOR", "").split(",")))   # gate | arm | leg (see _sensor_shortcut)
        self.bias_deg = float(os.environ.get("MOBILEPOSER_BIAS_DEG", 0))            # training-time constant sensor-mounting error (max angle, deg)
        self.bias_p = float(os.environ.get("MOBILEPOSER_BIAS_P", 0.5))
        self.ls_deg = float(os.environ.get("MOBILEPOSER_LS_DEG", 0))              # rotation label smoothing: noise std in degrees (0 = off)
        self.ls_fc = float(os.environ.get("MOBILEPOSER_LS_FC", 1.5))
        self.ls_all = os.environ.get("MOBILEPOSER_LS_ALL") == "1"
        if "hinge" in self.anchor:     # learnable joint axes: elbow = Rot(a0, flexion) Rot(a1, forearm roll), knee = Rot(a2, flexion); init: SMPL T-pose y / x / x
            self.hinge_axes = nn.Parameter(torch.tensor([[0., 1., 0.], [1., 0., 0.], [1., 0., 0.]]))
        n_in = self.C.n_output_joints*3 + self.C.n_imu + (N_EXTRA if self.relacc else 0) + (N_LEVER if self.lever else 0) + (N_INTEG if use_integ() else 0)
        self.token = os.environ.get("MOBILEPOSER_TOKEN") == "1"
        if self.token:      # part-wise discrete pose tokens (models/pose_tokenizer.py); K/nq must match the tokenizer file
            from mobileposer.models.pose_tokenizer import TokenPoseNet
            self.pose = TokenPoseNet(n_in, 256, K=int(os.environ.get("MOBILEPOSER_TOKEN_K", 1024)), nq=int(os.environ.get("MOBILEPOSER_TOKEN_NQ", 3)))
        else:
            self.pose = make_seq_model(n_in, joint_set.n_reduced*6 + (35 if "gate" in self.anchor else 0), 256, mode="poser") # pose estimation model
        self.token_w = float(os.environ.get("MOBILEPOSER_TOKEN_W", 0.02))
        self.marker_weight = float(os.environ.get("MOBILEPOSER_MARKER_LOSS", 0))
        if self.marker_weight > 0:
            self._init_markers()
        self.dir_weight = float(os.environ.get('MOBILEPOSER_DIRLOSS', 0))
        self.geo_weight = float(os.environ.get("MOBILEPOSER_GEOLOSS", 0))
        self.sip_w = float(os.environ.get("MOBILEPOSER_SIPW", 1.0))
        self.imu_cycle_weight = float(os.environ.get("MOBILEPOSER_IMU_CYCLE", 0.0))
        self.imu_cycle_fps = float(os.environ.get("MOBILEPOSER_FPS", datasets.fps))
        if self.imu_cycle_weight > 0:
            self.imu_cycle_labels = ["lw", "rw", "ls", "rs", "pelvis"]
            pnp_config = Path(__file__).resolve().parents[1] / "configs/pnp_surface_imu.json"
            self.imu_cycle_attachments = load_attachments(pnp_config, self.bodymodel)

        # loss function
        self.loss = nn.MSELoss()
        self.t_weight = 1e-5 
        self.use_pos_loss = True

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = Poser.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def _reduced_global_to_full(self, reduced_pose):
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        pose = reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
        pred_pose = self.global_to_local_pose(pose)
        for ignore in joint_set.ignored: pred_pose[:, ignore] = torch.eye(3, device=self.C.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose

    def _init_markers(self, n=96):
        v, w, J0 = self.bodymodel._v_template, self.bodymodel._skinning_weights, self.bodymodel._J
        idx = [int(torch.argmax(v[:, 1]))]
        dist = (v - v[idx[0]]).norm(dim=1)
        for _ in range(n - 1):
            idx.append(int(torch.argmax(dist)))
            dist = torch.minimum(dist, (v - v[idx[-1]]).norm(dim=1))
        idx = torch.tensor(idx)
        jid = w[idx].argmax(1)
        self.register_buffer("m_joint", jid, persistent=False)
        self.register_buffer("m_off", v[idx] - J0[jid], persistent=False)

    def forward(self, batch, input_lengths=None):
        # forward the pose prediction model
        batch = augment(batch, self.C.n_imu, self.relacc, self.lever)
        pred_pose, _, _ = self.pose(batch, input_lengths)
        if self.shortcut:
            pred_pose = self._sensor_shortcut(batch, pred_pose)
        return pred_pose

    def _sensor_shortcut(self, batch, pred):
        """Identity shortcut for the SENSED bones: their orientation is measured, so the network only predicts a (small) correction
        R_out = R_meas @ R(delta) instead of re-deriving the whole rotation.  Output slots (reduced joint order) <- sensor slots."""
        ns = self.C.n_joints
        B, S = pred.shape[:2]
        ori = batch[..., 72 + 3 * ns:72 + 12 * ns].reshape(B, S, ns, 3, 3)
        slots = {0: ns - 1, 4: 2, 5: 3, 14: 0, 15: 1}                      # pelvis <- last sensor, shanks (joints 4, 5), forearms (joints 18, 19)
        n_out = joint_set.n_reduced * 6
        extra = pred[..., n_out:]                                          # 'gate': 5 free absolute orientations (6D) + 5 gate logits
        out = list(pred[..., :n_out].reshape(B, S, -1, 6).unbind(2))
        id6 = art.math.rotation_matrix_to_r6d(torch.eye(3, device=pred.device)).reshape(6)
        eye = torch.eye(3, device=pred.device)
        for k, (oi, sl) in enumerate(slots.items()):
            Rm = ori[:, :, sl]
            Rm = torch.where((Rm.abs().sum((-1, -2)) > 0.5)[..., None, None], Rm, eye)            # zero-padded steps: identity (avoids NaN in the 6D normalisation)
            R = art.math.r6d_to_rotation_matrix((out[oi] + id6).reshape(-1, 6)).reshape(B, S, 3, 3)
            a6 = art.math.rotation_matrix_to_r6d((Rm @ R).reshape(-1, 3, 3)).reshape(B, S, 6)
            if 'gate' in self.anchor:
                # calibration-aware soft anchoring: blend the anchored (measurement x correction) and a freely predicted orientation with a learned,
                # per-bone, per-frame trust w; initialised to trust the measurement (logit + 3)
                w = torch.sigmoid(extra[..., 30 + k:31 + k] + 3.0)
                a6 = w * a6 + (1 - w) * extra[..., 6 * k:6 * k + 6]
            out[oi] = a6
        # kinematic-relative output for the UNSENSED parent bones: R_parent = R_child_out @ E^T, with E the (predicted) joint rotation
        # (elbow / knee). The upper arm / thigh orientation is tied to the measured forearm / shank orientation through a single joint rotation.
        rel = []
        if 'arm' in self.anchor or 'hinge' in self.anchor: rel += [(12, 14), (13, 15)]    # shoulders (joints 16, 17) <- forearms
        if 'leg' in self.anchor or 'hinge' in self.anchor: rel += [(1, 4), (2, 5)]        # hips (joints 1, 2) <- shanks
        for oi, ci in rel:
            if 'hinge' in self.anchor:
                # anatomically constrained joint rotation: elbow 2 DoF (flexion, roll), knee 1 DoF (flexion) about learnable fixed axes
                ax = torch.nn.functional.normalize(self.hinge_axes, dim=-1)
                def rot(a, t):                                                              # Rodrigues about the unit axis a; smooth at t = 0 (no norm / division)
                    z = torch.zeros((), device=a.device)
                    K = torch.stack([z, -a[2], a[1], a[2], z, -a[0], -a[1], a[0], z]).view(3, 3)
                    return torch.eye(3, device=a.device) + t.sin()[..., None, None] * K + (1 - t.cos())[..., None, None] * (K @ K)
                E = rot(ax[0], out[oi][..., 0]) @ rot(ax[1], out[oi][..., 1]) if oi in (12, 13) else rot(ax[2], out[oi][..., 0])
            else:
                E = art.math.r6d_to_rotation_matrix((out[oi] + id6).reshape(-1, 6)).reshape(B, S, 3, 3)
            Rc = art.math.r6d_to_rotation_matrix(out[ci].reshape(-1, 6)).reshape(B, S, 3, 3)
            out[oi] = art.math.rotation_matrix_to_r6d((Rc @ E.transpose(-1, -2)).reshape(-1, 3, 3)).reshape(B, S, 6)
        return torch.stack(out, dim=2).reshape(B, S, -1)

    @torch.no_grad()
    def _ls_noise(self, B, S, dev):
        """Smooth rotation-noise field [B,S,24,3] (rad): a low-frequency part shared along each kinematic chain plus a per-joint offset, built from a few
        random sinusoids below MOBILEPOSER_LS_FC Hz (0 = white noise). Std ~ MOBILEPOSER_LS_DEG degrees; the root is never perturbed."""
        amp = self.ls_deg * np.pi / 180
        if self.ls_fc <= 0:
            n = torch.randn(B, S, 24, 3, device=dev) * amp
        else:
            K, M = 4, 5 * 3 + 24 * 3
            t = torch.arange(S, device=dev).float() / float(datasets.fps)
            f = 0.1 + torch.rand(B, K, M, device=dev) * (self.ls_fc - 0.1)
            ph = torch.rand(B, K, M, device=dev) * 2 * np.pi
            sig = (torch.sin(2 * np.pi * f[:, None] * t[None, :, None, None] + ph[:, None]).sum(2)) * (2.0 / K) ** 0.5          # [B,S,M], unit variance
            base, off = sig[..., :15].view(B, S, 5, 3), sig[..., 15:].view(B, S, 24, 3)
            chain = torch.tensor([0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 2, 3, 4, 3, 4, 3, 4, 3, 4], device=dev)
            n = (0.8 * base[:, :, chain] + 0.6 * off) * amp
        n[:, :, 0] = 0
        return n

    @torch.no_grad()
    def _smooth_labels(self, tgt24):
        """Label smoothing in rotation space: perturb the LOCAL joint rotations with the smooth chain-correlated noise and recompose the global rotations
        (children follow their perturbed parents, so joint correlation holds by construction). Unless MOBILEPOSER_LS_ALL=1 the globally-measured
        (sensed) bones keep their exact labels, because the anchored output must reproduce them."""
        from mobileposer.models.pose_tokenizer import PARENT
        B, S = tgt24.shape[:2]
        R = art.math.r6d_to_rotation_matrix(tgt24.reshape(-1, 6)).view(B, S, 24, 3, 3)
        n = self._ls_noise(B, S, R.device)
        E = art.math.axis_angle_to_rotation_matrix(n.reshape(-1, 3)).view(B, S, 24, 3, 3)
        G = [R[:, :, 0]]
        for j in range(1, 24):
            loc = R[:, :, PARENT[j]].transpose(-1, -2) @ R[:, :, j]
            G.append(G[PARENT[j]] @ loc @ E[:, :, j])
        G = torch.stack(G, 2)
        if not self.ls_all:
            for j in (0, 4, 5, 18, 19):
                G[:, :, j] = R[:, :, j]
        return art.math.rotation_matrix_to_r6d(G.reshape(-1, 3, 3)).view(B, S, 24, 6)

    def _joint_noise(self, ref):
        """Corruption of the GT joint positions fed to the Poser during training (the Joints module supplies predictions at test time).
        MOBILEPOSER_JNOISE=std (default 0.04 m); MOBILEPOSER_JNOISE_W=w: temporally correlated noise (Gaussian-smoothed over w frames, unit variance kept), 0 = i.i.d."""
        std = float(os.environ.get("MOBILEPOSER_JNOISE", 0.04))
        w = int(os.environ.get("MOBILEPOSER_JNOISE_W", 0))
        noise = torch.randn(ref.size(), device=ref.device)
        if w > 0:
            k = torch.exp(-0.5 * (torch.arange(-3 * w, 3 * w + 1, device=ref.device).float() / w) ** 2)
            k = k / k.norm()
            D = ref.shape[-1]
            noise = F.conv1d(noise.transpose(1, 2), k.view(1, 1, -1).expand(D, 1, -1).contiguous(), padding=3 * w, groups=D).transpose(1, 2)
        return noise * std

    def _bias_aug(self, imu):
        """Constant per-window, per-sensor mounting error: ori' = ori @ R_b (angle ~ U(0, bias_deg), random axis), with probability bias_p per window.
        imu [B,S,60] = acc 15 | ori 45 (zero-padded steps stay zero)."""
        ns = self.C.n_joints
        B, S, _ = imu.shape
        ang = torch.rand(B, ns, device=imu.device) * self.bias_deg * np.pi / 180 * (torch.rand(B, 1, device=imu.device) < self.bias_p).float()
        axis = torch.nn.functional.normalize(torch.randn(B, ns, 3, device=imu.device), dim=-1)
        Rb = art.math.axis_angle_to_rotation_matrix((axis * ang[..., None]).reshape(-1, 3)).reshape(B, 1, ns, 3, 3)
        ori = imu[..., 3 * ns:12 * ns].reshape(B, S, ns, 3, 3) @ Rb
        return torch.cat([imu[..., :3 * ns], ori.reshape(B, S, -1)], -1)

    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, output_lengths = outputs

        # target pose
        target_pose = outputs['poses'] # [batch_size, window_length, 144] 
        B, S, _ = target_pose.shape

        # target joints
        joints = outputs['joints']
        target_joints = joints.view(B, S, -1)

        # generate noise for target joints for beter robustness
        noise = self._joint_noise(target_joints)
        noisy_joints = target_joints + noise

        # predict pose
        if self.bias_deg > 0 and self.training:
            imu_inputs = self._bias_aug(imu_inputs)
        pose_input = torch.cat((noisy_joints, imu_inputs), dim=-1)
        if self.token:
            self.pose.tf_targets = target_pose.view(B, S, 24, 6)    # teacher forcing for the autoregressive token mode
        pose_p = self(pose_input, input_lengths)

        # compute pose loss
        tgt24 = target_pose.view(B, S, 24, 6)
        if self.ls_deg > 0 and self.training:
            tgt24 = self._smooth_labels(tgt24)
        pose_t = tgt24[:, :, joint_set.reduced].reshape(B, S, -1)
        if self.sip_w != 1.0:
            # metric-aware weighting: SIP is the global orientation error of the hips and shoulders (reduced slots 1, 2, 12, 13)
            wj = torch.ones(joint_set.n_reduced, device=pose_p.device)
            wj[[1, 2, 12, 13]] = self.sip_w
            loss = ((pose_p - pose_t) ** 2 * wj.repeat_interleave(6)).mean() / wj.mean()
        elif self.shortcut:                                         # the shortcut outputs are valid rotations even on zero-padded steps: mask them
            vmask = (torch.arange(S, device=pose_p.device)[None] < torch.as_tensor(input_lengths, device=pose_p.device)[:, None]).float()
            loss = (((pose_p - pose_t) ** 2).mean(-1) * vmask).sum() / vmask.sum()
        else:
            loss = self.loss(pose_p, pose_t)
        loss += self.t_weight*self.compute_jerk_loss(pose_p)

        if self.token:
            lens = torch.as_tensor(input_lengths, device=pose_p.device)
            valid = torch.arange(S, device=pose_p.device)[None] < lens[:, None]
            loss += self.token_w * self.pose.token_loss(target_pose.view(B, S, 24, 6), valid)

        if self.geo_weight > 0:
            # geodesic (angle) loss on the reduced global rotations: matches the evaluation metric and, unlike MSE,
            # does not average multi-modal answers of the unsensed joints towards a mean
            lens = torch.as_tensor(input_lengths, device=pose_p.device)
            valid = torch.arange(S, device=pose_p.device)[None] < lens[:, None]
            Rp = art.math.r6d_to_rotation_matrix(pose_p[valid].reshape(-1, 6))
            Rt = art.math.r6d_to_rotation_matrix(pose_t[valid].reshape(-1, 6))
            cos = ((Rp.transpose(-1, -2) @ Rt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
            loss += self.geo_weight * torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).mean()

        # joint position loss
        if self.use_pos_loss:
            full_pose_p = self._reduced_global_to_full(pose_p)
            glob_p, joints_p = self.bodymodel.forward_kinematics(pose=full_pose_p.view(-1, 216))[:2]
            joints_p = joints_p.view(B, S, -1)
            if self.shortcut:
                vm = (torch.arange(S, device=pose_p.device)[None] < torch.as_tensor(input_lengths, device=pose_p.device)[:, None]).float()
                loss += (((joints_p - target_joints) ** 2).mean(-1) * vm).sum() / vm.sum()
            else:
                loss += self.loss(joints_p, target_joints)
            if self.marker_weight > 0:
                # surface-marker loss: 96 template vertices rigidly attached to their dominant bone; sensitive to the
                # bone orientation (incl. twist) and to where the limb surface is, not only to the joint centres
                Rp = glob_p.reshape(B, S, 24, 3, 3)
                Rt = art.math.r6d_to_rotation_matrix(target_pose.reshape(-1, 6)).view(B, S, 24, 3, 3)
                jp, jt = joints_p.view(B, S, 24, 3), target_joints.view(B, S, 24, 3)
                mp = jp[:, :, self.m_joint] + (Rp[:, :, self.m_joint] @ self.m_off.unsqueeze(-1)).squeeze(-1)
                mt = jt[:, :, self.m_joint] + (Rt[:, :, self.m_joint] @ self.m_off.unsqueeze(-1)).squeeze(-1)
                loss += self.marker_weight * self.loss(mp, mt)
            if self.dir_weight > 0:
                # bone-direction loss: ignores the (unobservable) axial twist, penalises where the limb points
                jp, jt = joints_p.view(B, S, 24, 3), target_joints.view(B, S, 24, 3)
                par = self.bodymodel.parent
                idx = list(range(1, 24))
                bp = jp[:, :, idx] - jp[:, :, [par[i] for i in idx]]
                bt = jt[:, :, idx] - jt[:, :, [par[i] for i in idx]]
                nt = bt.norm(dim=-1)
                cos = (bp * bt).sum(-1) / (bp.norm(dim=-1) * nt + 1e-6)
                valid = (nt > 1e-3).float()
                loss += self.dir_weight * ((1 - cos) * valid).sum() / valid.sum().clamp(min=1)

        if self.imu_cycle_weight > 0:
            local_pose = self._reduced_global_to_full(pose_p).reshape(B, S, 24, 3, 3)
            rendered_acc, _ = render_imu(
                self.bodymodel, local_pose, self.imu_cycle_labels,
                self.imu_cycle_attachments, self.imu_cycle_fps,
            )
            measured_acc = imu_inputs[..., :15].reshape(B, S, 5, 3)
            # Absolute translation acceleration is unobservable from the pose
            # head; subtracting pelvis acceleration removes that nuisance term.
            pred_rel = (rendered_acc[:, :, :4] - rendered_acc[:, :, 4:5]) / amass.acc_scale
            obs_rel = measured_acc[:, :, :4] - measured_acc[:, :, 4:5]
            lens = torch.as_tensor(input_lengths, device=pose_p.device)
            frame = torch.arange(S, device=pose_p.device)[None]
            valid = frame < lens[:, None]
            if S > 8:
                valid = valid & (frame >= 4) & (frame < (lens - 4)[:, None])
            cycle_err = F.smooth_l1_loss(pred_rel, obs_rel, reduction="none").mean((-1, -2))
            loss += self.imu_cycle_weight * (cycle_err * valid).sum() / valid.sum().clamp_min(1)

        return loss

    def compute_jerk_loss(self, pred_pose):
        jerk = pred_pose[:, 3:, :] - 3*pred_pose[:, 2:-1, :] + 3*pred_pose[:, 1:-2, :] - pred_pose[:, :-3, :]
        l1_norm = torch.norm(jerk, p=1, dim=2)
        return l1_norm.sum(dim=1).mean()

    def compute_temporal_loss(self, pred_pose):
        acc = pred_pose[:, 2:, :] + pred_pose[:, :-2, :] - 2*pred_pose[:, 1:-1, :]
        l1_norm = torch.norm(acc, p=1, dim=2)
        return l1_norm.sum(dim=1).mean() 

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(loss.item())
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("validation_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.validation_step_loss.append(loss.item())
        return {"loss": loss}
    
    def predict_step(self, batch, batch_idx):
        inputs, target = batch
        imu_inputs, input_lengths = inputs
        return self(imu_inputs, input_lengths)

    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()    # free memory

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()  # free memory

    def on_test_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="test")

    def epoch_end_callback(self, outputs, loop_type):
        # log average loss
        average_loss = torch.mean(torch.Tensor(outputs))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)
        # log learning late
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, prog_bar=True)

    def configure_optimizers(self):
        if arch() in ('tf', 'kin'):
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(os.environ.get('MOBILEPOSER_LR', 5e-4)), weight_decay=0.01)
            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 1000))
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hypers.lr)
        if os.environ.get("MOBILEPOSER_SCHED") == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hypers.num_epochs if hasattr(self.hypers, "num_epochs") else 60, eta_min=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return optimizer
