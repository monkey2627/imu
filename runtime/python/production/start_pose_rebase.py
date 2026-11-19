"""Rebase predicted/sensed motion deltas onto a prescribed exercise start pose."""
from __future__ import annotations

import numpy as np
import torch

from production.pose_calibration import LABEL_JOINT, SMPL_PARENTS
from production.start_pose_constraint import StartPoseConstraint, _globals


class StartPoseRebaser:
    """Use a fixed first-frame origin instead of temporarily holding a pose.

    The prescribed pose is built once from the selected start-pose template.
    Every later output is ``target_global_0 @ source_global_0.T @ source_global_t``.
    Sensed segments use calibrated IMU globals as their source; unobserved
    joints use the model globals.  Nothing is integrated recursively.
    """

    def __init__(self, labels):
        self.labels = list(labels)
        self.clear()

    def clear(self):
        self.action = None
        self.start_pose = None
        self.initial_source = None
        self.target_global = None

    def activate(self, action: str, start_pose: str):
        self.clear()
        self.action = action
        self.start_pose = start_pose

    @property
    def enabled(self):
        return self.action == "deadbug" and self.start_pose == "tabletop"

    def _source_globals(self, pose, measured):
        source = [value.copy() for value in _globals(pose)]
        for index, label in enumerate(self.labels):
            joint = LABEL_JOINT.get(label)
            if joint is not None:
                source[joint] = measured[index].copy()
        return source

    def apply(self, local, measured):
        is_torch = isinstance(local, torch.Tensor)
        pose = (local.detach().cpu().numpy().astype(np.float64) if is_torch
                else np.asarray(local, dtype=np.float64))
        diag = {"action": self.action, "startPose": self.start_pose,
                "enabled": self.enabled, "captured": self.initial_source is not None}
        if not self.enabled or measured is None:
            return local, diag

        measured = np.asarray(measured, dtype=np.float64)
        source = self._source_globals(pose, measured)
        if self.initial_source is None:
            template = StartPoseConstraint(self.labels)
            template.activate(self.action, self.start_pose)
            target_local, _ = template.apply(pose, measured, dt_seconds=0.0)
            target_local = np.asarray(target_local, dtype=np.float64)
            self.initial_source = [value.copy() for value in source]
            self.target_global = [value.copy() for value in _globals(target_local)]

        desired = [target @ initial.T @ current
                   for target, initial, current in
                   zip(self.target_global, self.initial_source, source)]
        out = np.empty_like(pose)
        for joint, parent in enumerate(SMPL_PARENTS):
            out[joint] = desired[joint] if parent < 0 else desired[parent].T @ desired[joint]

        diag["captured"] = True
        if is_torch:
            out = torch.from_numpy(out).to(local)
        return out, diag
