import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np

from mobileposer.config import *
import mobileposer.articulate as art
from mobileposer.models.rnn import RNN
from mobileposer.models.seq import make_seq_model, arch
from mobileposer.models.relacc import N_EXTRA, N_LEVER, N_INTEG, use_relacc, use_lever, use_integ, augment
from mobileposer.models.arm_nullspace import elbow_from_swivel


class Joints(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: 24 Joint positions. 
    """

    def __init__(self, finetune: bool=False):
        super().__init__()

        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = finetune_hypers if finetune else train_hypers

        # model definitions
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.relacc, self.lever = use_relacc(), use_lever()
        n_in = self.C.n_imu + (N_EXTRA if self.relacc else 0) + (N_LEVER if self.lever else 0) + (N_INTEG if use_integ() else 0)
        self.slowfast = os.environ.get("MOBILEPOSER_SLOWFAST") == "1"
        self.slow_dim = int(os.environ.get("MOBILEPOSER_SLOW_DIM", 128))
        self.slow_weight = float(os.environ.get("MOBILEPOSER_SLOW_WEIGHT", 0.2))
        self.slow_kernel = int(os.environ.get("MOBILEPOSER_SLOW_KERNEL", 601))
        if self.slow_kernel < 3 or self.slow_kernel % 2 == 0:
            raise ValueError("MOBILEPOSER_SLOW_KERNEL must be an odd integer >= 3")
        # Fixed Hann-windowed sinc FIR: the same 0-0.2 Hz low-pass is used in
        # training, fine-tuning, and held-out evaluation.
        taps = torch.arange(self.slow_kernel, dtype=torch.float32) - (self.slow_kernel - 1) / 2
        cutoff = float(os.environ.get("MOBILEPOSER_SLOW_CUTOFF", 0.2)) / float(datasets.fps)
        kernel = 2 * cutoff * torch.sinc(2 * cutoff * taps)
        kernel = kernel * torch.hann_window(self.slow_kernel, periodic=False)
        kernel = kernel / kernel.sum()
        self.register_buffer("slow_fir", kernel, persistent=False)
        # MOBILEPOSER_JCHAIN=1: bone-chain output. The network predicts, per bone, a unit direction and a bounded length scale; positions are
        # accumulated along the kinematic chain from the root with the template bone length (rigid-body structure built into the output).
        self.jchain = os.environ.get("MOBILEPOSER_JCHAIN") == "1"
        self.arm_nullspace = os.environ.get("MOBILEPOSER_ARM_NULLSPACE") == "1"
        if self.arm_nullspace and (self.jchain or self.slowfast or os.environ.get("MOBILEPOSER_PRIV") == "1"):
            raise ValueError("ARM_NULLSPACE currently requires plain Joints without JCHAIN, SLOWFAST, or PRIV")
        joint_dim = 24 * (4 if self.jchain else 3)
        # Four swivel coordinates plus a bounded 3-D elbow residual per arm.
        head_dim = joint_dim + (10 if self.arm_nullspace else 0)
        if self.slowfast:
            self.slow_encoder = make_seq_model(n_in, self.slow_dim, 256, mode="joints")
            self.slow_decoder = nn.Linear(self.slow_dim, joint_dim)
            self.fast_encoder = RNN(n_in, joint_dim, 128, n_rnn_layer=1)
            self.joints = None
        else:
            self.joints = make_seq_model(n_in, head_dim, 256, mode="joints") # joint estimation model
        if self.arm_nullspace:
            if not isinstance(self.joints, RNN) or self.joints.rnn.bidirectional:
                raise ValueError("ARM_NULLSPACE requires an explicitly causal RNN trunk (MOBILEPOSER_CAUSAL=1)")
            with torch.no_grad():
                self.joints.linear2.bias[72:76].copy_(self.joints.linear2.bias.new_tensor([0.0, 1.0, 0.0, 1.0]))
                self.joints.linear2.bias[76:82].zero_()
                self.joints.linear2.weight[76:82].zero_()
            J = self.bodymodel._J.detach().cpu().float()
            lengths = torch.tensor([
                [(J[16] - J[18]).norm(), (J[18] - J[20]).norm()],
                [(J[17] - J[19]).norm(), (J[19] - J[21]).norm()],
            ])
            self.register_buffer("arm_bone_lengths", lengths)
            self.register_buffer("arm_forearm_axes", torch.stack([
                F.normalize(J[20] - J[18], dim=0),
                F.normalize(J[21] - J[19], dim=0),
            ]))
            self.arm_length_log_scale = nn.Parameter(torch.zeros(2, 2))
            self.arm_reach_weight = float(os.environ.get("MOBILEPOSER_ARM_REACH_WEIGHT", 1.0))
            self.arm_swivel_weight = float(os.environ.get("MOBILEPOSER_ARM_SWIVEL_WEIGHT", 0.01))
            self.arm_orientation_weight = float(os.environ.get("MOBILEPOSER_ARM_ORIENTATION_WEIGHT", 0.1))
            self.arm_residual_limit = float(os.environ.get("MOBILEPOSER_ARM_RESIDUAL_LIMIT", 0.02))
            self.arm_residual_weight = float(os.environ.get("MOBILEPOSER_ARM_RESIDUAL_WEIGHT", 0.1))
            self.arm_bone_weight = float(os.environ.get("MOBILEPOSER_ARM_BONE_WEIGHT", 1.0))
        if self.jchain:
            from mobileposer.models.pose_tokenizer import PARENT
            J = self.bodymodel._J.detach().cpu().float()
            self.register_buffer("blen", torch.tensor([0.0] + [float((J[j] - J[PARENT[j]]).norm()) for j in range(1, 24)]), persistent=False)
        # MOBILEPOSER_PRIV=1: privileged-information distillation. A teacher with extra "virtual" IMUs on unsensed joints (orientation from
        # the GT pose, acceleration from the finite-differenced GT joint trajectory) is trained jointly; the 5-IMU student matches its
        # LSTM features (and outputs). Only the student is used at inference. MOBILEPOSER_PRIV_DISTILL=0 switches the teacher off (fine-tuning on real data).
        self.priv = os.environ.get("MOBILEPOSER_PRIV") == "1"
        self.priv_joints = [1, 2, 6, 9, 15, 16, 17]
        if self.priv:
            self.teacher = RNN(n_in + 12 * len(self.priv_joints), 24 * (4 if self.jchain else 3), 256)
            self.adapt = nn.Linear(512, 512)
        self.priv_on = os.environ.get("MOBILEPOSER_PRIV_DISTILL", "1") == "1"
        self.priv_lambda = float(os.environ.get("MOBILEPOSER_PRIV_LAMBDA", 1.0))

        # loss function
        self.loss = nn.MSELoss()
        self.t_weight = 1e-5

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = Joints.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def forward(self, batch, input_lengths=None):
        return self._predict(batch, input_lengths)[0]

    def _predict(self, batch, input_lengths=None):
        # forward joint model
        batch = augment(batch, self.C.n_imu, self.relacc, self.lever)
        if self.slowfast:
            joints, _ = self._slowfast(batch, input_lengths)
        else:
            joints, _, _ = self.joints(batch, input_lengths)
        if self.arm_nullspace:
            prediction, correction, swivel, orientation_error, residual, bone_error = self._decode_arm_nullspace(joints, batch)
            return prediction, {
                "reach_correction": correction,
                "swivel": swivel,
                "orientation_error": orientation_error,
                "residual": residual,
                "bone_error": bone_error,
            }
        return self._post(joints), None

    def _decode_arm_nullspace(self, raw, imu_inputs):
        B, S = raw.shape[:2]
        xyz = raw[..., :72].reshape(B, S, 24, 3).clone()
        swivel = raw[..., 72:76].reshape(B, S, 2, 2)
        residual = self.arm_residual_limit * torch.tanh(raw[..., 76:82].reshape(B, S, 2, 3))
        imu = imu_inputs[..., -60:]
        sensor_ori = imu[..., 15:60].reshape(B, S, 5, 3, 3)
        torso_up = xyz[:, :, 9] - xyz[:, :, 0]
        scales = torch.exp(0.05 * torch.tanh(self.arm_length_log_scale))
        corrections = []
        orientation_errors = []
        bone_errors = []
        for side, (shoulder, elbow, wrist) in enumerate(((16, 18, 20), (17, 19, 21))):
            lengths = self.arm_bone_lengths[side] * scales[side]
            elbow_pos, wrist_pos, correction = elbow_from_swivel(
                xyz[:, :, shoulder], xyz[:, :, wrist], lengths[0], lengths[1],
                swivel[:, :, side], torso_up,
            )
            elbow_pos = elbow_pos + residual[:, :, side]
            xyz[:, :, elbow] = elbow_pos
            xyz[:, :, wrist] = wrist_pos
            corrections.append(correction)
            # Wrist IMU orientation maps the template forearm axis into the
            # measured frame. A calibration-offset audit is still required
            # before applying this assumption to real captures.
            measured_axis = sensor_ori[:, :, side] @ self.arm_forearm_axes[side]
            decoded_axis = F.normalize(wrist_pos - elbow_pos, dim=-1, eps=1e-6)
            orientation_errors.append(1.0 - (decoded_axis * measured_axis).sum(-1).clamp(-1.0, 1.0))
            bone_errors.append(torch.stack([
                (xyz[:, :, shoulder] - elbow_pos).norm(dim=-1) - lengths[0],
                (wrist_pos - elbow_pos).norm(dim=-1) - lengths[1],
            ], dim=-1))
        return (
            xyz.reshape(B, S, 72),
            torch.stack(corrections, dim=-1),
            swivel,
            torch.stack(orientation_errors, dim=-1),
            residual,
            torch.stack(bone_errors, dim=-2),
        )

    def _slowfast(self, batch, input_lengths=None):
        slow_z, _, _ = self.slow_encoder(batch, input_lengths)
        slow_raw = self.slow_decoder(slow_z)
        fast_raw, _, _ = self.fast_encoder(batch, input_lengths)
        return slow_raw + fast_raw, slow_raw

    def _lowpass(self, x):
        k = self.slow_kernel
        y = F.pad(x.transpose(1, 2), (k // 2, k // 2), mode="replicate")
        kernel = self.slow_fir.to(dtype=x.dtype, device=x.device).view(1, 1, k)
        y = F.conv1d(y.reshape(-1, 1, y.shape[-1]), kernel, stride=1)
        return y.reshape(x.shape[0], x.shape[2], x.shape[1]).transpose(1, 2)

    @staticmethod
    def _valid_mask(lengths, steps, device):
        lens = torch.as_tensor(lengths, device=device)
        return torch.arange(steps, device=device)[None, :] < lens[:, None]

    def _post(self, raw):
        """Network output -> [B,S,72] root-relative joint positions (chain accumulation for MOBILEPOSER_JCHAIN)."""
        if not self.jchain:
            return raw
        from mobileposer.models.pose_tokenizer import PARENT
        B, S, _ = raw.shape
        r = raw.view(B, S, 24, 4)
        d = F.normalize(r[..., :3], dim=-1, eps=1e-6)
        L = self.blen * (1 + 0.3 * torch.tanh(r[..., 3]))
        pos = [raw.new_zeros(B, S, 3)]
        for j in range(1, 24):
            pos.append(pos[PARENT[j]] + L[:, :, j, None] * d[:, :, j])
        return torch.stack(pos, 2).view(B, S, 72)

    def _priv_inputs(self, outputs):
        """Virtual IMUs on self.priv_joints from the GT: [B,S,12*n] = accelerations (world, /acc_scale) then orientations."""
        B, S = outputs['joints'].shape[:2]
        P = outputs['joints'].reshape(B, S, 24, 3)[:, :, self.priv_joints] + outputs['trans'].reshape(B, S, 1, 3)
        acc = torch.zeros_like(P)
        acc[:, 1:-1] = (P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]) * float(datasets.fps) ** 2
        acc = (acc / amass.acc_scale).clamp(-10, 10)
        R = art.math.r6d_to_rotation_matrix(outputs['poses'].reshape(-1, 6)).view(B, S, 24, 9)[:, :, self.priv_joints]
        return torch.cat([acc.reshape(B, S, -1), R.reshape(B, S, -1)], -1)

    @staticmethod
    def _run(rnn, x, lens):
        """RNN forward that also returns the bidirectional LSTM features (before the output layer)."""
        from torch.nn.functional import relu
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        data = rnn.dropout(relu(rnn.linear1(x)))
        data = pack_padded_sequence(data, lens, batch_first=True, enforce_sorted=False)
        data, _ = rnn.rnn(data)
        data, _ = pad_packed_sequence(data, batch_first=True, total_length=x.shape[1])
        return rnn.linear2(data), data

    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, _ = outputs

        # target joints
        joints = outputs['joints']
        target_joints = joints.view(joints.shape[0], joints.shape[1], -1)

        # predicted joints
        distill = self.priv and self.priv_on and self.training and not self.slowfast and not self.arm_nullspace
        arm_aux = None
        if distill:
            s_raw, s_feat = self._run(self.joints, augment(imu_inputs, self.C.n_imu, self.relacc, self.lever), input_lengths)
            pred_joints = self._post(s_raw)
            t_raw, t_feat = self._run(self.teacher, torch.cat([imu_inputs, self._priv_inputs(outputs)], -1), input_lengths)
            t_joints = self._post(t_raw)
        else:
            if self.slowfast:
                raw, slow_raw = self._slowfast(augment(imu_inputs, self.C.n_imu, self.relacc, self.lever), input_lengths)
                pred_joints = self._post(raw)
            else:
                pred_joints, arm_aux = self._predict(imu_inputs, input_lengths)

        # compute loss
        loss = self.loss(pred_joints, target_joints)
        if self.slowfast:
            slow_pred = self._post(slow_raw)
            slow_gt = outputs.get('slow_joints', self._lowpass(target_joints)).to(target_joints.device).reshape_as(target_joints)
            mask = self._valid_mask(input_lengths, target_joints.shape[1], target_joints.device).unsqueeze(-1)
            loss = loss + self.slow_weight * (slow_pred - slow_gt).pow(2).masked_select(mask).mean()
        if distill:
            loss = loss + self.loss(t_joints, target_joints)
            loss = loss + self.priv_lambda * self.loss(self.adapt(s_feat), t_feat.detach()) + 0.5 * self.loss(pred_joints, t_joints.detach())
        if arm_aux is not None:
            valid = self._valid_mask(input_lengths, target_joints.shape[1], target_joints.device)
            reach = arm_aux["reach_correction"]
            reach_mask = valid.unsqueeze(-1).expand_as(reach)
            reach_loss = reach.square().masked_select(reach_mask).mean()
            swivel = F.normalize(arm_aux["swivel"], dim=-1, eps=1e-6)
            continuity = 1 - (swivel[:, 1:] * swivel[:, :-1]).sum(-1)
            pair_mask = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1).expand_as(continuity)
            swivel_loss = continuity.masked_select(pair_mask).mean() if pair_mask.any() else continuity.sum() * 0
            orientation = arm_aux["orientation_error"]
            orientation_loss = orientation.masked_select(valid.unsqueeze(-1).expand_as(orientation)).mean()
            residual = arm_aux["residual"]
            residual_mask = valid[:, :, None, None].expand_as(residual)
            residual_loss = residual.square().masked_select(residual_mask).mean()
            bone_error = arm_aux["bone_error"]
            bone_mask = valid[:, :, None, None].expand_as(bone_error)
            bone_loss = bone_error.square().masked_select(bone_mask).mean()
            loss = (
                loss
                + self.arm_reach_weight * reach_loss
                + self.arm_swivel_weight * swivel_loss
                + self.arm_orientation_weight * orientation_loss
                + self.arm_residual_weight * residual_loss
                + self.arm_bone_weight * bone_loss
            )
        loss += self.t_weight*self.compute_temporal_loss(pred_joints)
        return loss

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
        self.training_step_loss.clear()

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hypers.lr)
        if os.environ.get("MOBILEPOSER_SCHED") == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hypers.num_epochs if hasattr(self.hypers, "num_epochs") else 60, eta_min=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return optimizer
