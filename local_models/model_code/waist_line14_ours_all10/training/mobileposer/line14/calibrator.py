"""Inference adapter for a completed pre-prediction C20 calibration clip."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from mobileposer.line14.causal import CalibrationState
from mobileposer.line14.link import LinkEncoder
from mobileposer.line9.enc2 import phys_features
from mobileposer.line9.robust_calib import heading_normalize


class C20Calibrator:
    """Estimate frozen sensor-to-bone calibration from one finished clip.

    ``imu`` must follow C20's packed ``acc/30 | orientation`` convention and
    already be sampled at the checkpoint's rate (the standard C20 pipeline is
    30 Hz, every second source frame). Phase IDs use the trained 0..7 vocab.
    This class only consumes calibration data; target-motion frames belong to
    :class:`CalibratedCausalSession` after this object returns.
    """

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cpu", stride: int = 2):
        self.device = torch.device(device)
        self.stride = int(stride)
        if self.stride != 2:
            raise ValueError("the audited C20 uni checkpoint requires source stride=2 (30 Hz calibration input)")
        path = Path(checkpoint)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("mode") != "uni" or payload.get("cas"):
            raise ValueError("C20Calibrator currently requires a unified (uni), non-cascade checkpoint")
        if not all(k in payload for k in ("model", "fm", "fs", "ym", "ys")):
            raise ValueError("checkpoint is missing model or normalization statistics")
        weights = payload["model"]
        hidden = int(weights["proj.0.weight"].shape[0])
        in_dim = int(weights["proj.0.weight"].shape[1])
        latent = int(weights["norm.weight"].numel())
        max_phases = int(weights["phase_emb.weight"].shape[0])
        layers = 1 + max(int(m.group(1)) for key in weights
                         if (m := re.fullmatch(r"gru\.weight_ih_l(\d+)", key)))
        if in_dim != 135 or payload["ym"].numel() != 24 or payload["ys"].numel() != 24:
            raise ValueError("checkpoint dimensions do not match C20 phys-feature / R2 outputs")
        self.model = LinkEncoder(in_dim, geo_dim=24, with_rot=True, with_geo=True,
                                 hidden=hidden, layers=layers, latent=latent,
                                 heads=4, max_phases=max_phases).to(self.device)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.fm = torch.as_tensor(payload["fm"], dtype=torch.float32, device=self.device)
        self.fs = torch.as_tensor(payload["fs"], dtype=torch.float32, device=self.device)
        self.ym = torch.as_tensor(payload["ym"], dtype=torch.float32, device=self.device)
        self.ys = torch.as_tensor(payload["ys"], dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def estimate(self, imu: Tensor, phase_ids: Tensor, mask: Optional[Tensor] = None) -> CalibrationState:
        """Analyze a complete, independent calibration clip and freeze its estimates."""
        imu = torch.as_tensor(imu, dtype=torch.float32, device=self.device)
        phase_ids = torch.as_tensor(phase_ids, dtype=torch.long, device=self.device)
        if imu.ndim != 2 or imu.shape[1] != 60 or len(imu) < 2:
            raise ValueError("calibration imu must have shape [T>=2,60]")
        if phase_ids.shape != (len(imu),):
            raise ValueError("phase_ids must have shape [T]")
        if not torch.isfinite(imu).all() or (phase_ids < 0).any() or (phase_ids >= 8).any():
            raise ValueError("calibration inputs must be finite with phase IDs in [0,7]")
        valid = torch.ones_like(phase_ids, dtype=torch.bool) if mask is None else torch.as_tensor(
            mask, dtype=torch.bool, device=self.device
        )
        if valid.shape != phase_ids.shape or not valid.any():
            raise ValueError("mask must have shape [T] and at least one valid frame")

        batch_imu = heading_normalize(imu.unsqueeze(0))
        features = phys_features(batch_imu, dt=self.stride / 60.0)
        normalized = (features - self.fm) / self.fs.clamp_min(1e-4)
        normalized = normalized * valid.view(1, -1, 1)
        geo, rotation = self.model(
            normalized,
            phase_ids.unsqueeze(0),
            valid.unsqueeze(0),
        )
        mean_cm = geo[0, :24] * self.ys + self.ym
        logvar_cm = geo[0, 24:].clamp(-6, 4) + 2.0 * torch.log(self.ys.clamp_min(1e-6))
        return CalibrationState.from_link_output(rotation[0], mean_cm, logvar_cm)
