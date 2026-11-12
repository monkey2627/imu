import os
import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import random
import lightning as L
from tqdm import tqdm 

import mobileposer.articulate as art
from mobileposer.config import *
from mobileposer.utils import *
from mobileposer.helpers import *


def lowpass_full_sequence(x, fps=None, cutoff=0.2, kernel_size=601):
    """Apply the shared FIR low-pass before any windowing."""
    fps = int(fps or datasets.fps)
    taps = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2
    kernel = 2 * (cutoff / fps) * torch.sinc(2 * (cutoff / fps) * taps)
    kernel = kernel * torch.hann_window(kernel_size, periodic=False, device=x.device, dtype=x.dtype)
    kernel = kernel / kernel.sum()
    shape = x.shape
    y = F.pad(x.reshape(1, -1, shape[0]).transpose(1, 2), (kernel_size // 2, kernel_size // 2), mode="replicate")
    y = F.conv1d(y.reshape(-1, 1, y.shape[-1]), kernel.view(1, 1, -1))
    return y.reshape(shape[0], shape[1], shape[2])


class PoseDataset(Dataset):
    def __init__(self, fold: str='train', evaluate: str=None, finetune: str=None):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.combos = list(amass.combos.items())
        self._ctx = None
        self.data = self._prepare_dataset()

    def _get_data_files(self, data_folder):
        if self.fold == 'train':
            return self._get_train_files(data_folder)
        elif self.fold == 'test':
            return self._get_test_files()
        else:
            raise ValueError(f"Unknown data fold: {self.fold}.")

    def _get_train_files(self, data_folder):
        if self.finetune:
            return [datasets.finetune_datasets[self.finetune]]
        holdout = os.environ.get("MOBILEPOSER_DATASET_NAMES")
        if holdout:
            names = {name.strip() + ".pt" for name in holdout.split(",") if name.strip()}
            available = {x.name for x in data_folder.iterdir() if not x.is_dir()}
            missing = names - available
            if missing:
                raise FileNotFoundError(f"requested dataset files not found in {data_folder}: {sorted(missing)}")
            return sorted(names)
        return [x.name for x in data_folder.iterdir() if not x.is_dir()]

    def _get_test_files(self):
        return [datasets.test_datasets[self.evaluate]]

    def _prepare_dataset(self):
        cache = os.environ.get("MOBILEPOSER_DATA_CACHE")
        if cache and os.path.exists(cache) and not (self.finetune or self.evaluate):
            print(f"loading cached windows: {cache}", flush=True)
            return torch.load(cache)
        data = self._prepare_dataset_uncached()
        if cache and not (self.finetune or self.evaluate):
            torch.save(data, cache + ".tmp")
            os.replace(cache + ".tmp", cache)
        return data

    def _prepare_dataset_uncached(self):
        data_folder = paths.processed_datasets / ('eval' if (self.finetune or self.evaluate) else '')
        data_files = self._get_data_files(data_folder)
        data = {key: [] for key in ['imu_inputs', 'pose_outputs', 'joint_outputs', 'slow_joint_outputs', 'tran_outputs', 'vel_outputs', 'foot_outputs']}
        for data_file in tqdm(data_files):
            try:
                file_data = torch.load(data_folder / data_file)
                self._process_file_data(file_data, data)
            except Exception as e:
                print(f"Error processing {data_file}: {e}.")
        return data

    def _process_file_data(self, file_data, data):
        accs, oris, poses, trans = file_data['acc'], file_data['ori'], file_data['pose'], file_data['tran']
        joints = file_data.get("joint", [None] * len(poses))
        shapes = file_data.get("shape", [None] * len(poses))
        foots = file_data.get('contact', [None] * len(poses))

        for acc, ori, pose, tran, joint, foot, shp in zip(accs, oris, poses, trans, joints, foots, shapes):
            if os.environ.get("MOBILEPOSER_CTX") == "1":
                from mobileposer.body_ctx import body_context
                self._ctx = body_context(shp if shp is not None else torch.zeros(10))
            ns = model_config.n_joints
            acc, ori = acc[:, :ns]/amass.acc_scale, ori[:, :ns]
            pose_global, joint = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216)) # convert local rotation to global
            pose = pose if self.evaluate else pose_global.view(-1, 24, 3, 3)                # use global only for training
            joint = joint.view(-1, 24, 3)
            slow_joint = lowpass_full_sequence(joint, datasets.fps, float(os.environ.get("MOBILEPOSER_SLOW_CUTOFF", 0.2)), int(os.environ.get("MOBILEPOSER_SLOW_KERNEL", 601)))
            lag = int(os.environ.get('MOBILEPOSER_LAG', 0))
            if lag > 0 and not self.evaluate and len(acc) > lag + 12:
                # Experiment: fixed-lag causal output. The target of frame t is predicted from IMU up to t+lag
                # (the frame of the sensors is the current one, t+lag).
                acc, ori = acc[lag:], ori[lag:]
                pose, joint, slow_joint, tran = pose[:-lag], joint[:-lag], slow_joint[:-lag], tran[:-lag]
                foot = foot[:-lag] if foot is not None else None
            from mobileposer.frames import frame_kind, frame_slot, frame_rot
            if frame_kind() != "world" and not self.evaluate:
                # Experiment: express everything in the WAIST sensor's frame (slot 4 of the five-IMU layouts).
                # Inputs: acc/ori of every sensor rotated by R_waist^T; targets: global rotations and root-relative
                # joint positions rotated likewise. Frame is per-frame (time-varying), so only Poser/Joints are trained.
                RwT = frame_rot(ori[:, frame_slot()], frame_kind()).transpose(-1, -2)   # [T,3,3]
                acc = (RwT.unsqueeze(1) @ acc.unsqueeze(-1)).squeeze(-1)           # [T,5,3]
                ori = RwT.unsqueeze(1) @ ori                                       # [T,5,3,3]
                pose = RwT.unsqueeze(1) @ pose                                     # [T,24,3,3]
                joint = (RwT.unsqueeze(1) @ joint.unsqueeze(-1)).squeeze(-1)       # [T,24,3]
            self._process_combo_data(acc, ori, pose, joint, slow_joint, tran, foot, data)

    def _process_combo_data(self, acc, ori, pose, joint, slow_joint, tran, foot, data):
        for _, c in self.combos:
            # mask out layers for different subsets
            combo_acc = torch.zeros_like(acc)
            combo_ori = torch.zeros_like(ori)
            combo_acc[:, c] = acc[:, c]
            combo_ori[:, c] = ori[:, c]
            imu_input = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1) # [[N, 15], [N, 45]] => [N, 60] 
            if self._ctx is not None:
                imu_input = torch.cat([imu_input, self._ctx.unsqueeze(0).expand(len(imu_input), -1)], dim=1)   # + body context columns

            data_len = len(imu_input) if self.evaluate or os.environ.get("MOBILEPOSER_FULL_SEQUENCES") == "1" else datasets.window_length
            
            for key, value in zip(['imu_inputs', 'pose_outputs', 'joint_outputs', 'slow_joint_outputs', 'tran_outputs'],
                                [imu_input, pose, joint, slow_joint, tran]):
                data[key].extend(torch.split(value, data_len))

            if not (self.evaluate or self.finetune): # do not finetune translation module
                self._process_translation_data(joint, tran, foot, data_len, data)

    def _process_translation_data(self, joint, tran, foot, data_len, data):
        root_vel = torch.cat((torch.zeros(1, 3), tran[1:] - tran[:-1]))
        vel = torch.cat((torch.zeros(1, 24, 3), torch.diff(joint, dim=0)))
        vel[:, 0] = root_vel
        data['vel_outputs'].extend(torch.split(vel * (datasets.fps / amass.vel_scale), data_len))
        data['foot_outputs'].extend(torch.split(foot, data_len))

    def __getitem__(self, idx):
        imu = self.data['imu_inputs'][idx].float()
        sigma = float(os.environ.get('MOBILEPOSER_OFFSET_DEG', 0))
        if sigma > 0 and not (self.evaluate or self.finetune):
            # Experiment: per-window constant sensor-to-bone orientation offsets (as in real use, where they are
            # unknown). Orientations get a random constant rotation per sensor; accelerations (world frame) and
            # targets stay untouched, so the model must learn to be robust to / partially undo the offsets.
            S = imu.shape[0]
            axis = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
            ang = torch.randn(5).abs() * (sigma * np.pi / 180)
            C = art.math.axis_angle_to_rotation_matrix(axis * ang.unsqueeze(-1)).view(5, 3, 3)
            ori = imu[:, 15:60].reshape(S, 5, 3, 3) @ C
            imu = torch.cat([imu[:, :15], ori.reshape(S, 45)], dim=1)
        if os.environ.get("MOBILEPOSER_NOISE") == "1" and not (self.evaluate or self.finetune):
            # Domain-standard IMU noise model applied to the synthetic training input (as in DIP / TransPose / PNP-style synthesis):
            # constant per-sensor orientation calibration error, per-frame orientation jitter, accelerometer white noise, bias and gain error.
            ns, S = model_config.n_joints, imu.shape[0]
            ori_sd = float(os.environ.get("MOBILEPOSER_NOISE_ORI_DEG", 5)) * np.pi / 180
            jit_sd = float(os.environ.get("MOBILEPOSER_NOISE_JIT_DEG", 1)) * np.pi / 180
            acc_sd = float(os.environ.get("MOBILEPOSER_NOISE_ACC", 0.3)) / 30
            bias_sd = float(os.environ.get("MOBILEPOSER_NOISE_BIAS", 0.2)) / 30
            gain_sd = float(os.environ.get("MOBILEPOSER_NOISE_GAIN", 0.1))
            acc = imu[:, :3 * ns].reshape(S, ns, 3)
            ori = imu[:, 3 * ns:3 * ns + 9 * ns].reshape(S, ns, 3, 3)
            axis = torch.nn.functional.normalize(torch.randn(ns, 3), dim=-1)
            C = art.math.axis_angle_to_rotation_matrix(axis * (torch.randn(ns).abs() * ori_sd).unsqueeze(-1)).view(ns, 3, 3)
            Jt = art.math.axis_angle_to_rotation_matrix((torch.randn(S * ns, 3) * jit_sd)).view(S, ns, 3, 3)
            ori = Jt @ ori @ C
            acc = acc * (1 + gain_sd * torch.randn(1, ns, 1)) + bias_sd * torch.randn(1, ns, 3) + acc_sd * torch.randn(S, ns, 3)
            imu = torch.cat([acc.reshape(S, 3 * ns), ori.reshape(S, 9 * ns), imu[:, 12 * ns:]], dim=1)
        if os.environ.get("MOBILEPOSER_CALERR") and not (self.evaluate or self.finetune):
            # line 10: structured calibration / orientation-estimation error on the synthetic inputs (labels unchanged); see calib_error.py. Do not combine with NOISE / OFFSET_DEG / BIAS_DEG.
            from mobileposer.calib_error import corrupt
            imu = corrupt(imu, os.environ["MOBILEPOSER_CALERR"], datasets.fps)
        if os.environ.get("MOBILEPOSER_CTX") == "1" and imu.shape[1] > 60 and not (self.evaluate or self.finetune):
            # body size / lever arms come from an image-based estimate at test time: simulate its error (one common scale + per-value noise)
            sd = float(os.environ.get("MOBILEPOSER_CTX_NOISE", 0.04))
            imu = imu.clone()
            imu[:, 60:] = imu[:, 60:] * (1 + sd * torch.randn(1)) * (1 + 0.5 * sd * torch.randn(imu.shape[1] - 60))
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()
        num_pred_joints = len(amass.pred_joints_set)
        pose = art.math.rotation_matrix_to_r6d(self.data['pose_outputs'][idx]).reshape(-1, num_pred_joints, 6)[:, amass.pred_joints_set].reshape(-1, 6*num_pred_joints)

        if self.evaluate:
            return imu, pose, joint, tran
        if self.finetune:
            slow = self.data.get('slow_joint_outputs')
            return (imu, pose, joint, tran, slow[idx].float()) if slow is not None else (imu, pose, joint, tran)

        vel = self.data['vel_outputs'][idx].float()
        contact = self.data['foot_outputs'][idx].float()

        slow = self.data.get('slow_joint_outputs')
        slow = slow[idx].float() if slow is not None and len(slow) else joint
        return imu, pose, joint, tran, vel, contact, slow

    def __len__(self):
        return len(self.data['imu_inputs'])

def pad_seq(batch):
    """Pad sequences to same length for RNN."""
    def _pad(sequence):
        padded = nn.utils.rnn.pad_sequence(sequence, batch_first=True)
        lengths = [seq.shape[0] for seq in sequence]
        return padded, lengths

    def _fields(item):
        imu, pose, joint, tran = item[:4]
        if len(item) >= 7:
            slow = item[6]
        elif len(item) >= 5 and item[4].shape[-1] == 3:
            slow = item[4]
        else:
            slow = joint
        return imu, pose, joint, slow, tran

    inputs, poses, joints, slow_joints, trans = zip(*[_fields(item) for item in batch])
    inputs, input_lengths = _pad(inputs)
    poses, pose_lengths = _pad(poses)
    joints, joint_lengths = _pad(joints)
    slow_joints, slow_joint_lengths = _pad(slow_joints)
    trans, tran_lengths = _pad(trans)
    outputs = {'poses': poses, 'joints': joints, 'slow_joints': slow_joints, 'trans': trans}
    output_lengths = {'poses': pose_lengths, 'joints': joint_lengths, 'slow_joints': slow_joint_lengths, 'trans': tran_lengths}

    if len(batch[0]) >= 7:
        vels, foots = zip(*[(item[4], item[5]) for item in batch])
        foot_contacts, foot_contact_lengths = _pad(foots)
        outputs['foot_contacts'], output_lengths['foot_contacts'] = foot_contacts, foot_contact_lengths
        vels, vel_lengths = _pad(vels)
        outputs['vels'], output_lengths['vels'] = _pad(vels), vel_lengths

    return (inputs, input_lengths), (outputs, output_lengths)


class PoseDataModule(L.LightningDataModule):
    def __init__(self, finetune: str = None):
        super().__init__()
        self.finetune = finetune
        self.hypers = finetune_hypers if self.finetune else train_hypers

    def setup(self, stage: str):
        if stage == 'fit':
            dataset = PoseDataset(fold='train', finetune=self.finetune)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            self.test_dataset = PoseDataset(fold='test', finetune=self.finetune)

    def _dataloader(self, dataset):
        return DataLoader(
            dataset, 
            batch_size=self.hypers.batch_size, 
            collate_fn=pad_seq, 
            num_workers=self.hypers.num_workers, 
            shuffle=True, 
            drop_last=True
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)
