"""Local WebSocket broadcaster: streams per-frame SMPL joint quaternions to
Unity, replacing SlimeVR's VMC output as the pose source for the avatar.

Plain JSON over WebSocket -- not SolarXR's flatbuffers protocol. This is a
new, project-local channel between our own two endpoints (this bridge and
the Unity receiver script), so there is no reason to take on flatbuffers
complexity here; SlimeVR-Server's own VMC/OSC output is untouched and can
keep running in parallel as a comparison reference (plan M6).
"""
from __future__ import annotations

import asyncio
import json
from typing import Set

import numpy as np
import websockets

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 21200  # arbitrary, distinct from SlimeVR-Server's 21110


class PoseBroadcaster:
    """Holds the set of connected Unity clients and fans out each frame."""

    def __init__(self) -> None:
        self._clients: Set = set()
        self.commands: asyncio.Queue[str] = asyncio.Queue()
        # Recording/replay requests from Unity, e.g. {"command": "startReplay", "path": "..."}.
        self.controls: asyncio.Queue[dict] = asyncio.Queue()

    async def _handler(self, websocket) -> None:
        self._clients.add(websocket)
        recording = False
        try:
            async for raw in websocket:
                try:
                    command = json.loads(raw)
                    model = command.get("selectModel") if isinstance(command, dict) else None
                    if isinstance(model, str) and model:
                        await self.commands.put(model)
                    if isinstance(command, dict) and isinstance(command.get("command"), str):
                        await self.controls.put(command)
                        if command["command"] in ("startRecording", "stopRecording"):
                            recording = command["command"] == "startRecording"
                except (TypeError, ValueError):
                    continue
        finally:
            self._clients.discard(websocket)
            # A Unity client that leaves Play mode mid-recording must not leave the file open.
            if recording:
                await self.controls.put({"command": "stopRecording", "reason": "client disconnected"})

    async def serve(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        return await websockets.serve(self._handler, host, port)

    def latest_model_command(self) -> str | None:
        latest = None
        while True:
            try:
                latest = self.commands.get_nowait()
            except asyncio.QueueEmpty:
                return latest

    def drain_controls(self) -> list[dict]:
        controls = []
        while True:
            try:
                controls.append(self.controls.get_nowait())
            except asyncio.QueueEmpty:
                return controls

    async def broadcast(self, frame_index: int, layout_name: str, joints_xyzw: np.ndarray,
                        contact: np.ndarray | None = None,
                        tracker_rotations: np.ndarray | None = None,
                        diagnostics: dict | None = None,
                        comparison: dict | None = None,
                        model_name: str | None = None) -> None:
        """joints_xyzw: (24, 4) array of (x, y, z, w) local-rotation quaternions,
        already converted to Unity's left-handed convention (see
        rotmat.batch_smpl_matrix_to_unity_quat_xyzw), in mobileposer's fixed
        24-joint SMPL order (0=pelvis, 1/2=hip, ..., see mobileposer/fit_ours_smpl.py's
        joint-index comment for the full list).

        "joints" is sent as a flat 96-float array (24 * 4), not a nested
        array-of-arrays: Unity's JsonUtility (used on the receiving end, see
        Assets/slimeVR/Scripts/MobilePoserPoseSource.cs) cannot deserialize
        nested JSON arrays, only flat arrays of a primitive type.
        """
        if not self._clients:
            return
        message = {
            "frame": frame_index,
            "layout": layout_name,
            "joints": joints_xyzw.reshape(-1).tolist(),
        }
        if contact is not None:
            message["footContact"] = np.asarray(contact, dtype=np.float32).reshape(-1).tolist()
        if tracker_rotations is not None:
            message["trackerRotations"] = np.asarray(tracker_rotations, dtype=np.float32).reshape(-1).tolist()
        if diagnostics:
            message.update(diagnostics)
        if comparison:
            message["comparison"] = comparison
        if model_name:
            message["model"] = model_name
        payload = json.dumps(message)
        stale = []
        for client in list(self._clients):   # a client may disconnect while we await its send
            try:
                await client.send(payload)
            except (websockets.ConnectionClosed, ConnectionAbortedError, ConnectionResetError, OSError):
                stale.append(client)
        for client in stale:
            self._clients.discard(client)
