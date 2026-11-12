"""Frozen calibration-error sampler for line 14 stage C (error injection on real motions).

What it models: the error of the sensor-to-bone ORIENTATION-OFFSET estimate of the rot estimator, for ONE wearing (= one calibration, aggregated over
`ns` calibration sessions the way the downstream pipeline aggregates them: matrix mean projected to SO(3)).  Fitted on out-of-fold (OOF) AMASS identities only.

Convention (the same as scripts_run/line14_build_cal_*.py and line14_assemble_link.py):  measured orientation = R_bone @ R_off ; the pipeline corrects with R_meas @ R_est^T.
The error is the LEFT tangent vector  delta = Log(R_est R_true^T)  (axis-angle, radians) per sensor, order lw, rw, ls, rs, pelvis, axes = those of the stored rotation
matrices.  Injection:  R_est = Exp(delta) @ R_oracle  (stays in SO(3)).  A wearing gets ONE delta that is constant over the whole motion (never resampled per frame).

Draws are deterministic functions of (identity/wearing key, Monte-Carlo index, sampler version, ns): Gaussian N(0, Sigma) in the 15-d tangent space (zero mean; the fold-0
bias did not reproduce on the other folds), truncated by rejection at the chi2_15 0.999 radius.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch

SENSORS = ("lw", "rw", "ls", "rs", "pelvis")
CHI2_15_999 = 37.6978          # chi2.ppf(0.999, 15)


def _rng(key: str, mc: int, version: str, ns: int) -> np.random.Generator:
    h = hashlib.sha256(f"{key}|{mc}|{version}|{ns}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(h[:8], "little")))


def load_sampler(path: str) -> dict:
    return torch.load(path, weights_only=False)


def draw_delta(sampler: dict, key: str, mc: int = 0, ns: int = 8) -> torch.Tensor:
    """[5,3] radians, deterministic in (key, mc, version, ns)."""
    spec = sampler["ns"][ns]
    L = spec["L"].numpy()                                   # [15,15], Sigma = L L^T (radians)
    rng = _rng(key, mc, sampler["version"], ns)
    for _ in range(100):
        z = rng.standard_normal(15)
        if float(z @ z) <= CHI2_15_999:                     # whitened coordinates: ||z||^2 is the Mahalanobis distance
            break
    return torch.as_tensor(L @ z, dtype=torch.float32).reshape(5, 3)


def apply_error(R_oracle: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """R_oracle [5,3,3] (the oracle sensor-to-bone offsets in the stored convention), delta [5,3] -> R_est = Exp(delta) @ R_oracle."""
    from mobileposer.articulate import math as M
    E = M.axis_angle_to_rotation_matrix(delta.reshape(-1, 3).float()).reshape(5, 3, 3)
    return E @ R_oracle.float()
