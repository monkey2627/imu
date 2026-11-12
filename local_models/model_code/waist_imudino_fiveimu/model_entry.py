"""Online adapter: calibrated world IMU -> articulated SMPL + measured waist root."""
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .downstream.data import to_imu6
from .downstream.model import FiveIMUHead, encode_imu
from .imudino import IMUFormer

MODEL_ID = "waist_imudino_fiveimu"
SENSOR_ORDER = ["lw", "rw", "ls", "rs", "waist"]
# Newest usable head index: slots 472..479 are the clamped plateau of the
# align_corners=False patch interpolation, and one extra real sample finalizes gyro.
NEWEST_INDEX = 471
MIN_LAG = 480 - NEWEST_INDEX  # 9
DEFAULT_LAG = 15
MAX_LAG = 48


def configured_lag(value=None):
    """Output lag in 60 Hz frames (MOTION_RECOVER_IMUDINO_LAG); 9 = newest window only."""
    lag = int(value if value is not None else os.environ.get("MOTION_RECOVER_IMUDINO_LAG", DEFAULT_LAG))
    if not MIN_LAG <= lag <= MAX_LAG:
        raise ValueError(f"IMUDINO lag must be in [{MIN_LAG}, {MAX_LAG}] frames, got {lag}")
    return lag


def sixd_to_matrix(values):
    """Exact column convention from downstream/render_validation._sixd_to_matrix."""
    first, second = values[..., :3], values[..., 3:6]
    column1 = F.normalize(first, dim=-1, eps=1e-8)
    second = second - (column1 * second).sum(-1, keepdim=True) * column1
    column2 = F.normalize(second, dim=-1, eps=1e-8)
    column3 = torch.cross(column1, column2, dim=-1)
    return torch.stack((column1, column2, column3), dim=-1)


class FiveIMUAdapter:
    """Online read-out of the window-trained head.

    The head was trained/validated on fixed 480-frame windows (stride 240, overlap
    averaged). Its prediction for one physical frame depends on where that frame
    sits in the window: re-reading only the newest index of a window that slides
    by 1-3 frames turns this placement variance into frame-to-frame jumps
    (recording 20261111_235722: same frame, windows 2 frames apart, L-knee p99 18 deg,
    while a single window's own trajectory moves only ~5 deg p99 over 2 frames).

    Output for frame t = newest-lag is therefore the 6D average of every already
    computed window that saw t at index <= 471, re-orthogonalized like the
    validation stitcher. No neighbouring frames are mixed and no extra inference
    is run; lag=9 reproduces the previous newest-window-only behaviour.
    """
    window = 480
    fps = 60

    def __init__(self, checkpoint: Path, device=None, lag=None):
        self.lag = configured_lag(lag)
        requested = device or os.environ.get("MOTION_RECOVER_IMUDINO_DEVICE", "auto")
        self.device = torch.device("cuda:0" if requested == "auto" and torch.cuda.is_available()
                                   else "cpu" if requested == "auto" else requested)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state.get("sensor_order") != SENSOR_ORDER or state.get("window_frames") != self.window
                or state.get("target") != "SMPL local 6D joints 1..23"):
            raise ValueError("IMUDINO checkpoint does not match the five-IMU deployment contract")
        self.backbone = IMUFormer().eval()
        self.backbone.load_state_dict(torch.load(checkpoint.parent / "imudino.pt",
                                                map_location="cpu", weights_only=True), strict=True)
        self.head = FiveIMUHead().eval()
        self.head.load_state_dict(state["head"], strict=True)
        self.backbone.requires_grad_(False).to(self.device)
        self.head.requires_grad_(False).to(self.device)
        self.orientations = deque(maxlen=self.window + 1)
        self.accelerations = deque(maxlen=self.window + 1)
        self.pushed = 0          # absolute count of pushed frames (frame ids)
        self.votes = {}          # frame id -> [sum of 6D [138], window count]

    def prime(self, feature, orientation, acceleration):
        self.orientations.clear()
        self.accelerations.clear()
        self.votes.clear()
        for _ in range(self.window):
            self.push(feature, orientation, acceleration)

    def push(self, feature, orientation, acceleration):
        ori = np.asarray(orientation, dtype=np.float32)
        acc = np.asarray(acceleration, dtype=np.float32)
        if ori.shape != (5, 3, 3) or acc.shape != (5, 3):
            raise ValueError("IMUDINO requires exactly lw,rw,ls,rs,waist orientations and accelerations")
        if not np.isfinite(ori).all() or not np.isfinite(acc).all():
            raise ValueError("Non-finite IMUDINO observation")
        self.orientations.append(ori.copy())
        self.accelerations.append(acc.copy())
        self.pushed += 1

    @torch.inference_mode()
    def predict_window(self, acceleration, orientation):
        """Raw SI world inputs [480,5,3]/[480,5,3,3] -> local 6D [1,480,138]."""
        if len(acceleration) != self.window or len(orientation) != self.window:
            raise ValueError("Expected a complete 480-frame window")
        return self.predict_imu(to_imu6(acceleration, orientation, self.fps))

    @torch.inference_mode()
    def predict_imu(self, imu):
        """Encode finalized continuous-stream gyro, then decode the trained head."""
        if imu.shape != (self.window, 5, 6):
            raise ValueError("Expected finalized IMU tensor [480,5,6]")
        imu = imu[None].to(self.device)
        mask = torch.ones((1, self.window), dtype=torch.bool, device=self.device)
        # Keep float32 for portability; caches used CUDA bf16 then float16 storage.
        features = encode_imu(self.backbone, imu, mask, self.fps)
        return self.head(features, frames=self.window).float().cpu()

    def infer(self):
        if not self.orientations:
            raise RuntimeError("No IMUDINO observations")
        # Use one additional real observation to finalize every gyro sample in
        # the 480-frame window. The training function copies its final gyro;
        # discard that extra terminal sample before encoding instead.
        all_ori = torch.from_numpy(np.stack(self.orientations))
        all_acc = torch.from_numpy(np.stack(self.accelerations))
        padding = self.window + 1 - len(all_ori)
        if padding:
            all_ori = torch.cat((all_ori[:1].expand(padding, -1, -1, -1), all_ori))
            all_acc = torch.cat((all_acc[:1].expand(padding, -1, -1), all_acc))
        imu = to_imu6(all_acc, all_ori, self.fps)[:-1]
        prediction = self.predict_imu(imu)[0]
        # Window index i is frame id (newest - 480 + i); newest is the gyro lookahead frame.
        newest = self.pushed - 1
        first_index = self.window - self.lag
        target = newest - self.lag
        for index in range(first_index, NEWEST_INDEX + 1):
            frame = newest - self.window + index
            vote = self.votes.get(frame)
            if vote is None:
                self.votes[frame] = [prediction[index].double().clone(), 1]
            else:
                vote[0] += prediction[index].double()
                vote[1] += 1
        for frame in [f for f in self.votes if f < target]:
            del self.votes[frame]
        total, count = self.votes[target]
        self.last_vote_count = count
        target_index = first_index
        children = sixd_to_matrix((total / count).float().reshape(23, 6))
        if not torch.isfinite(children).all() or (torch.linalg.det(children) < 0.99).any():
            raise RuntimeError("IMUDINO predicted a degenerate rotation")
        # WAIST measures SMPL spine1 (joint 3), whose parent is pelvis (joint 0).
        # Therefore R_root @ R_local[3] = measured R_waist; no pseudo-label is needed.
        root = all_ori[target_index, 4] @ children[2].T
        return torch.cat((root[None], children))


def load(checkpoint=None, device=None):
    checkpoint = checkpoint or Path(__file__).resolve().parents[2] / "checkpoints" / MODEL_ID / "best.pt"
    return FiveIMUAdapter(Path(checkpoint), device=device)
