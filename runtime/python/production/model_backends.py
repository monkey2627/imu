"""Adapters for the waist-surface baseline checkpoints.

The three baselines were trained with different output heads.  This module
keeps those details out of the SolarXR loop and gives it one contract:
``step(feature, orientation, acceleration) -> (24, 3, 3)``.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from production.paths import paths
from mobileposer.line9.models import Line9Joints
from mobileposer.mobile_export import MobilePose, _gram_schmidt
from mobileposer.stream.features import imu_features, to_local_pose
from mobileposer.stream.model import StreamPoser

if TYPE_CHECKING:
    from production.calibration import HeadingCalibration
    from production.layout import ResolvedLayout

WAIST_MODEL_FILES = {
    "waist_mobileposer_amass": ("mobileposer", "waist_mobileposer_amass/base_model.pth"),
    "waist_mobileposer_deadbug": ("mobileposer", "waist_mobileposer_deadbug/model_finetuned.pth"),
    "waist_mobileposer_ours_all10": ("mobileposer", "waist_mobileposer_ours_all10/model_finetuned.pth"),
    "waist_mobileposer_ours_deadbug": ("mobileposer", "waist_mobileposer_ours_deadbug/model_specialized.pth"),
    "waist_streamposer_amass": ("stream", "waist_streamposer_amass/model.pt"),
    "waist_streamposer_deadbug": ("stream", "waist_streamposer_deadbug/model.pt"),
    "waist_streamposer_ours_all10": ("stream", "waist_streamposer_ours_all10/model_finetuned.pt"),
    "waist_streamposer_ours_deadbug": ("stream", "waist_streamposer_ours_deadbug/model_specialized.pt"),
    "waist_line14_amass": ("line14", "waist_line14_amass/ori-none_geom-none_s42.pt"),
    "waist_line14_deadbug": ("line14", "waist_line14_deadbug/model.pt"),
    "waist_line14_ours_all10": ("line14", "waist_line14_ours_all10/model_finetuned.pt"),
    "waist_line14_ours_deadbug": ("line14", "waist_line14_ours_deadbug/model_specialized.pt"),
    # Fine-tuned on Ours with each capture's sensor-vs-label offsets removed (see OURS_LABEL_WAIST_OFFSET)
    # plus live-matched mount/heading augmentation (train_*mountaug.py).
    "waist_mobileposer_ours_offsetfree": ("mobileposer", "waist_mobileposer_ours_offsetfree/model_finetuned.pth"),
    "waist_streamposer_ours_offsetfree": ("stream", "waist_streamposer_ours_offsetfree/model_finetuned.pt"),
    "waist_line14_ours_offsetfree": ("line14", "waist_line14_ours_offsetfree/model_finetuned.pt"),
    # Experimental (2026-11-06): StreamPoser + explicit sensor position/velocity stage (mobileposer/stream/posacc.py),
    # AMASS + the coach's Ours motions as clean synthetic IMU.  Did not learn to use the acceleration; see its README.
    "waist_streamposer_posacc": ("posacc", "waist_streamposer_posacc/model.pt"),
    "waist_imudino_fiveimu": ("imudino", "waist_imudino_fiveimu/best.pt"),
    # 2026-11-12: stateful streaming head, sensors at wrist/ankle straps (forearm/shank 95%), waist unchanged.
    "waist_imudino_stream_wristankle": ("imudino_stream", "waist_imudino_stream_wristankle/best.pt"),
    # Same head + BirdDogMix limb-recombination augmentation (decouples arm reach from leg lift).
    "waist_imudino_stream_wristankle_limbmix": ("imudino_stream", "waist_imudino_stream_wristankle_limbmix/best.pt"),
}
WAIST_MODEL_LABELS = {
    "waist_mobileposer_amass": "MobilePose / AMASS",
    "waist_mobileposer_deadbug": "MobilePose / Deadbug",
    "waist_mobileposer_ours_all10": "MobilePose / Ours 10-action",
    "waist_mobileposer_ours_deadbug": "MobilePose / Deadbug tuned",
    "waist_streamposer_amass": "StreamPoser / AMASS",
    "waist_streamposer_deadbug": "StreamPoser / Deadbug",
    "waist_streamposer_ours_all10": "StreamPoser / Ours 10-action",
    "waist_streamposer_ours_deadbug": "StreamPoser / Deadbug tuned",
    "waist_line14_amass": "Line14 / AMASS",
    "waist_line14_deadbug": "Line14 / Deadbug",
    "waist_line14_ours_all10": "Line14 / Ours 10-action",
    "waist_line14_ours_deadbug": "Line14 / Deadbug tuned",
    "waist_mobileposer_ours_offsetfree": "MobilePose / Ours offset-free",
    "waist_streamposer_ours_offsetfree": "StreamPoser / Ours offset-free",
    "waist_line14_ours_offsetfree": "Line14 / Ours offset-free",
    "waist_streamposer_posacc": "StreamPoser+PosAcc / AMASS+Ours synth (experimental)",
    "waist_imudino_fiveimu": "IMUDINO / Five IMU (AMASS + BirdDog)",
    "waist_imudino_stream_wristankle": "IMUDINO Stream / wrist+ankle",
    "waist_imudino_stream_wristankle_limbmix": "IMUDINO Stream / wrist+ankle limb-mix",
}
WAIST_CHECKPOINT_ROOT = paths.waist_checkpoint

# The offset-free fine-tunes saw the waist in the HumanPose-label convention: the SlimeVR waist
# orientation times this constant body-frame rotation (rotvec -17.3, -9.6, +1.5 deg; identical
# within 1.6 deg across all 10 Ours captures). Forearm/shank offsets vary per wearing, so they are
# left to the live calibration and the training-time mount augmentation.
OURS_LABEL_WAIST_OFFSET = np.array([[0.98584, -0.00047, -0.16767],
                                    [0.05032, 0.95474, 0.29315],
                                    [0.15994, -0.29744, 0.94125]], dtype=np.float32)


def input_orientation_offsets(model_id: str) -> dict:
    """Per-label right-multiplied rotations a model expects on its input orientations."""
    return {"waist": OURS_LABEL_WAIST_OFFSET} if "offsetfree" in model_id else {}


def apply_input_offsets(model_id: str, labels, ori: torch.Tensor) -> torch.Tensor:
    offsets = input_orientation_offsets(model_id)
    if not offsets:
        return ori
    ori = ori.clone()
    for label, rotation in offsets.items():
        if label in labels:
            index = labels.index(label)
            ori[index] = ori[index] @ torch.from_numpy(rotation).to(ori)
    return ori


def resolve_waist_model(model_id: str, checkpoint_root: Path = WAIST_CHECKPOINT_ROOT):
    spec = WAIST_MODEL_FILES.get(model_id)
    if spec is None:
        return None
    kind, relative = spec
    checkpoint = checkpoint_root / relative
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{model_id} checkpoint not found: {checkpoint}")
    if kind == "mobileposer":
        model = load_mobileposer_checkpoint(checkpoint)
        return kind, checkpoint, model
    return kind, checkpoint, load_waist_adapter(kind, checkpoint)


class WaistInferenceSession:
    def __init__(self, layout: ResolvedLayout, adapter, heading: HeadingCalibration):
        self.layout, self.adapter, self.heading = layout, adapter, heading

    def prepare_feature(self, quats_xyzw, accel_xyz):
        from production.calibration import bone_orientation
        ori = bone_orientation(quats_xyzw, self.layout.labels)
        accel = self.heading.apply(accel_xyz)
        acc = (accel.reshape(-1) / 30.0).numpy()
        feature = np.concatenate((acc, ori.reshape(-1).numpy())).astype(np.float32)
        return feature, ori, accel

    def prime(self, quats_xyzw, accel_xyz):
        feature, ori, accel = self.prepare_feature(quats_xyzw, accel_xyz)
        self.adapter.prime(feature, ori.numpy(), accel.numpy())

    def push_observation(self, feature, ori, accel):
        self.adapter.push(feature, ori.numpy(), accel.numpy())

    def push_feature(self, feature):
        raise RuntimeError("waist baseline requires orientation and acceleration samples")

    def infer(self):
        return self.adapter.infer()

    def finish_replay(self):
        if hasattr(self.adapter, "finish_replay"):
            self.adapter.finish_replay()


def make_waist_session(model_id, resolved, heading, checkpoint_root=WAIST_CHECKPOINT_ROOT, loaded=None):
    kind, checkpoint, model_or_adapter = loaded or resolve_waist_model(model_id, checkpoint_root)
    if kind == "mobileposer":
        from production.inference import InferenceSession
        return InferenceSession(layout=resolved, model=model_or_adapter, heading=heading)
    if kind in ("imudino", "imudino_stream") and list(resolved.labels) != ["lw", "rw", "ls", "rs", "waist"]:
        raise ValueError("IMUDINO requires sensor order lw,rw,ls,rs,waist")
    return WaistInferenceSession(resolved, model_or_adapter, heading)


def _imudino_entry(module="model_entry"):
    import sys
    code_root = str(paths.workspace_root / "local_models" / "model_code")
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    import importlib
    return importlib.import_module(f"waist_imudino_fiveimu.{module}")


def model_lag_frames(model_id):
    if model_id.startswith("waist_imudino_stream_"):
        return _imudino_entry("stream_entry").LAG
    if model_id == "waist_imudino_fiveimu":
        # Same-frame multi-window read-out; MOTION_RECOVER_IMUDINO_LAG (9..48, default 15).
        return _imudino_entry().configured_lag()
    return 5 if "streamposer" in model_id else 9 if "line14" in model_id else 10


class ChainPoser(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, 256)
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True)
        self.linear2 = nn.Linear(512, 96)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x, _ = self.rnn(x)
        return self.linear2(x)


def _full_local_from_reduced(r6d: torch.Tensor) -> torch.Tensor:
    """Convert Line14 reduced global rotations to full SMPL local rotations."""
    reduced = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    identity = torch.eye(3, dtype=r6d.dtype, device=r6d.device)
    global_rot = [identity for _ in range(24)]
    reduced_rot = _gram_schmidt(r6d.reshape(-1, 6)).reshape(16, 3, 3)
    for i, joint in enumerate(reduced):
        global_rot[joint] = reduced_rot[i]
    local = [global_rot[0]]
    for joint in range(1, 24):
        parent = parents[joint]
        if joint in {7, 8, 10, 11, 20, 21, 22, 23}:
            local.append(identity)
        else:
            local.append(global_rot[parent].transpose(-1, -2) @ global_rot[joint])
    return torch.stack(local)


class WaistStreamAdapter:
    def __init__(self, checkpoint: Path, window: int = 120):
        config = __import__("json").loads((checkpoint.parent / "config.json").read_text())
        self.lag = int(config.get("lag", 5))
        self.window = max(window, self.lag + 8)
        self.features = deque(maxlen=self.window)
        self.orientations = deque(maxlen=self.window)
        self.accelerations = deque(maxlen=self.window)
        self.pending_orientations = []
        self.pending_accelerations = []
        self.previous_orientation = None
        self.previous_acceleration = None
        self.state = None
        self.model = StreamPoser(
            gyro=bool(config.get("gyro", 1)), hidden=int(config.get("hidden", 512)),
            layers=int(config.get("layers", 3)), anchor=config.get("anchor", "none"),
            frame=config.get("frame", "both"), arch=config.get("arch", "mp2"),
            local=bool(config.get("local_gyro", 0)),
        )
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.eval()

    def prime(self, feature, orientation, acceleration):
        for _ in range(self.window):
            self.push(feature, orientation, acceleration)
        self._warm_state()

    def push(self, feature, orientation, acceleration):
        orientation = np.asarray(orientation, dtype=np.float32)
        acceleration = np.asarray(acceleration, dtype=np.float32)
        self.features.append(np.asarray(feature, dtype=np.float32))
        self.orientations.append(orientation)
        self.accelerations.append(acceleration)
        self.pending_orientations.append(orientation)
        self.pending_accelerations.append(acceleration)

    def _warm_state(self):
        if not self.pending_orientations:
            return
        ori_np = np.asarray(self.pending_orientations, dtype=np.float32)
        acc_np = np.asarray(self.pending_accelerations, dtype=np.float32)
        ori = torch.from_numpy(ori_np)[None]
        acc = torch.from_numpy(acc_np)[None]
        x = imu_features(acc, ori, gyro=True, frame=self.model.frame, local=self.model.local)
        with torch.no_grad():
            _, _, self.state = self.model(x, self.state)
        self.previous_orientation, self.previous_acceleration = ori_np[-1], acc_np[-1]
        self.pending_orientations.clear()
        self.pending_accelerations.clear()

    def infer(self):
        if not self.pending_orientations:
            self.push(self.features[-1], self.orientations[-1], self.accelerations[-1])
        new_ori = np.asarray(self.pending_orientations, dtype=np.float32)
        new_acc = np.asarray(self.pending_accelerations, dtype=np.float32)
        if self.previous_orientation is None:
            ori_np, acc_np = new_ori, new_acc
        else:
            ori_np = np.concatenate((self.previous_orientation[None], new_ori), axis=0)
            acc_np = np.concatenate((self.previous_acceleration[None], new_acc), axis=0)
        n_new = len(new_ori)
        ori = torch.from_numpy(ori_np)[None]
        acc = torch.from_numpy(acc_np)[None]
        x = imu_features(acc, ori, gyro=True, frame=self.model.frame, local=self.model.local)[:, -n_new:]
        ori_new = torch.from_numpy(new_ori)[None]
        with torch.no_grad():
            _, raw, self.state = self.model(x, self.state)
        root = torch.from_numpy(np.asarray(self.orientations, dtype=np.float32))[..., 4, :, :]
        root_frame = max(0, len(root) - 1 - self.lag)
        reduced = self.model.compose(raw, ori_new[:, :, 4].transpose(-1, -2).unsqueeze(2) @ ori_new[:, :, :4])
        target_root_frame = min(root_frame, max(0, len(root) - 1))
        output = to_local_pose(_body_model(), reduced[0, -1:], root[target_root_frame:target_root_frame + 1])[-1]
        self.previous_orientation, self.previous_acceleration = new_ori[-1], new_acc[-1]
        self.pending_orientations.clear()
        self.pending_accelerations.clear()
        return output

    def finish_replay(self):
        self._warm_state()


class WaistPosAccAdapter(WaistStreamAdapter):
    """PosAccPoser (mobileposer/stream/posacc.py) behind the stream adapter's contract: same buffers, lag
    and streaming logic, a different network.  The sensor-motion stage starts from an unknown pose."""

    def __init__(self, checkpoint: Path, window: int = 120):
        from mobileposer.stream.posacc import PosAccPoser
        config = __import__("json").loads((checkpoint.parent / "config.json").read_text())
        self.lag = int(config.get("lag", 5))
        self.window = max(window, self.lag + 8)
        self.features = deque(maxlen=self.window)
        self.orientations = deque(maxlen=self.window)
        self.accelerations = deque(maxlen=self.window)
        self.pending_orientations = []
        self.pending_accelerations = []
        self.previous_orientation = None
        self.previous_acceleration = None
        self.state = None
        self.model = PosAccPoser(gyro=bool(config.get("gyro", 1)), hidden=int(config.get("hidden", 512)),
                                 layers=int(config.get("layers", 3)), frame=config.get("frame", "both"))
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.eval()


class WaistLine14Adapter:
    def __init__(self, checkpoint: Path, window: int = 125):
        self.window = window
        self.features = deque(maxlen=window)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.joints = Line9Joints("b0", geom_dim=1)
        self.poser = ChainPoser(132)
        self.joints.load_state_dict(state["jnet"], strict=True)
        self.poser.load_state_dict(state["poser"], strict=True)
        self.joints.eval(); self.poser.eval()

    def prime(self, feature, orientation, acceleration):
        for _ in range(self.window): self.features.append(np.asarray(feature, dtype=np.float32))

    def push(self, feature, orientation, acceleration):
        self.features.append(np.asarray(feature, dtype=np.float32))

    def infer(self):
        x = torch.from_numpy(np.asarray(self.features, dtype=np.float32))[None]
        with torch.no_grad():
            joints = self.joints(x)
            reduced = self.poser(torch.cat((joints, x), dim=-1))[0, -10]
        return _full_local_from_reduced(reduced)


_BODY = None
def _body_model():
    global _BODY
    if _BODY is None:
        from mobileposer.articulate.model import ParametricModel
        _BODY = ParametricModel(paths.smpl_file, device="cpu")
    return _BODY


def load_waist_adapter(kind: str, checkpoint: Path):
    if kind == "imudino":
        return _imudino_entry().FiveIMUAdapter(checkpoint)
    if kind == "imudino_stream":
        return _imudino_entry("stream_entry").StreamAdapter(checkpoint)
    if kind == "stream": return WaistStreamAdapter(checkpoint)
    if kind == "posacc": return WaistPosAccAdapter(checkpoint)
    if kind == "line14": return WaistLine14Adapter(checkpoint)
    raise ValueError(kind)


def load_mobileposer_checkpoint(checkpoint: Path) -> MobilePose:
    model = MobilePose(anchor=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for module, prefix in ((model.joints, "joints.joints."), (model.pose, "pose.pose.")):
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not selected:
            raise ValueError(f"{checkpoint} has no {prefix} weights")
        module.load_state_dict(selected, strict=True)
    return model.eval()
