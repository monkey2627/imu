"""Online adapter for the stateful IMUDINO stream head (wrist/ankle placement).

push() only stores samples; infer() advances every pending frame in order, so the
schedule is identical to training (stream.common): token for grid frame g
(g % 16 == 15) from the 480-frame window ending at g, usable from frame g+2; the
GRU steps once per frame. Frame f is decoded when frame f+1 has arrived (forward
gyro), so the output lag is 1 frame. No output smoothing.
"""
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .downstream.data import to_imu6
from .downstream.model import encode_imu
from .imudino import IMUFormer
from .model_entry import sixd_to_matrix
from .stream.common import PATCH, TOKEN_DELAY, TOKEN_DIM, WINDOW, frame_features, token_slot
from .stream.model import StreamHead

MODEL_ID = "waist_imudino_stream_wristankle"
SENSOR_ORDER = ["lw", "rw", "ls", "rs", "waist"]
LAG = 1
MAX_CATCHUP = 600


class StreamAdapter:
    window = WINDOW
    fps = 60
    lag = LAG

    def __init__(self, checkpoint: Path, device=None):
        requested = device or os.environ.get("MOTION_RECOVER_IMUDINO_DEVICE", "auto")
        self.device = torch.device("cuda:0" if requested == "auto" and torch.cuda.is_available()
                                   else "cpu" if requested == "auto" else requested)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("sensor_order") != SENSOR_ORDER or state.get("contract") != "stream.common v1":
            raise ValueError("checkpoint does not match the stream.common v1 five-IMU contract")
        self.backbone = IMUFormer().eval()
        self.backbone.load_state_dict(torch.load(checkpoint.parent / "imudino.pt", map_location="cpu",
                                                 weights_only=True), strict=True)
        self.head = StreamHead().eval()
        self.head.load_state_dict(state["head"], strict=True)
        self.backbone.requires_grad_(False).to(self.device)
        self.head.requires_grad_(False).to(self.device)
        self.reset()

    def reset(self):
        # Enough history for a full token window even when infer() trails push() by MAX_CATCHUP frames.
        self.ori = deque(maxlen=self.window + MAX_CATCHUP + 4)   # frame ids pushed-len .. pushed-1
        self.acc = deque(maxlen=self.window + MAX_CATCHUP + 4)
        self.pushed = 0
        self.decoded = -1          # newest decoded frame id
        self.state = None
        self.token = torch.zeros(TOKEN_DIM)
        self.output = None

    def prime(self, feature, orientation, acceleration):
        self.reset()
        for _ in range(self.window):
            self.push(feature, orientation, acceleration)

    def push(self, feature, orientation, acceleration):
        ori = np.asarray(orientation, dtype=np.float32)
        acc = np.asarray(acceleration, dtype=np.float32)
        if ori.shape != (5, 3, 3) or acc.shape != (5, 3) or not (np.isfinite(ori).all() and np.isfinite(acc).all()):
            raise ValueError("stream adapter needs finite lw,rw,ls,rs,waist orientation [5,3,3] and accel [5,3]")
        self.ori.append(ori.copy())
        self.acc.append(acc.copy())
        self.pushed += 1

    def _history(self, first, last):
        """Stored frames [first, last] as tensors (ids must still be in the deques)."""
        base = self.pushed - len(self.ori)
        o = torch.from_numpy(np.stack([self.ori[i - base] for i in range(first, last + 1)]))
        a = torch.from_numpy(np.stack([self.acc[i - base] for i in range(first, last + 1)]))
        return o, a

    @torch.inference_mode()
    def _token(self, g):
        base = self.pushed - len(self.ori)
        lo = max(g + 1 - self.window, base)
        o, a = self._history(lo, g + 1)              # + lookahead frame for forward gyro
        imu = to_imu6(a, o, self.fps)[:-1]
        window = imu.new_zeros((self.window, 5, 6))
        mask = torch.zeros(self.window, dtype=torch.bool)
        window[self.window - len(imu):] = imu
        mask[self.window - len(imu):] = True
        tokens = encode_imu(self.backbone, window[None].to(self.device), mask[None].to(self.device), self.fps)
        return tokens[0, -1].float().cpu()  # float32 end to end, same as the training cache

    @torch.inference_mode()
    def infer(self):
        newest = self.pushed - 2               # newest frame whose gyro is final
        if newest < 0:
            raise RuntimeError("stream adapter needs at least two frames")
        start = max(self.decoded + 1, newest - MAX_CATCHUP + 1)
        if start > self.decoded + 1:
            self.state = None                  # gap: restart recurrent state like a new stream
            slot = int(token_slot(torch.tensor(start)))
            g0 = slot * PATCH + PATCH - 1
            self.token = self._token(g0) if slot >= 0 and g0 + 1 >= self.pushed - len(self.ori)                 else torch.zeros(TOKEN_DIM)
        for f in range(start, newest + 1):
            g = f - TOKEN_DELAY
            if g >= PATCH - 1 and g % PATCH == PATCH - 1:
                self.token = self._token(g)
            o, a = self._history(f, f + 1)
            imu = to_imu6(a, o, self.fps)[:1]
            feat = frame_features(imu, o[:1], torch.tensor([f]))
            pred, self.state = self.head(self.token[None, None].to(self.device),
                                         feat[None].to(self.device), self.state)
            self.output = (pred[0, 0].float().cpu(), o[0, 4])
            self.decoded = f
        sixd, waist = self.output
        children = sixd_to_matrix(sixd.reshape(23, 6))
        if not torch.isfinite(children).all():
            raise RuntimeError("stream head produced a degenerate rotation")
        root = waist @ children[2].T          # waist measures spine1: R_root @ R_local[3] = R_waist
        return torch.cat((root[None], children))


def load(checkpoint=None, device=None):
    checkpoint = checkpoint or Path(__file__).resolve().parents[2] / "checkpoints" / MODEL_ID / "best.pt"
    return StreamAdapter(Path(checkpoint), device=device)
