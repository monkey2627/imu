"""Record raw SolarXR tracker input from Unity and replay it into the live loop.

Unlike ``recording.py`` (a lossless diagnostics dump of features and model
outputs), this stores only what the realtime loop consumes -- the per-label
TrackerSample at every 60 Hz advance plus the upright reference -- so a replay
goes back through the *currently selected* model and model switching keeps
working without wearing the IMUs.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from production.paths import paths
from production.solarxr_client import TrackerSample

FORMAT = "mobileposer-unity-input"
FORMAT_VERSION = 1
RECORDINGS_DIR = paths.root_dir / "recordings"


def resolve_recording_path(value: str | None) -> Path:
    """Empty -> newest recording; bare names resolve inside RECORDINGS_DIR."""
    text = (value or "").strip().strip('"')
    if not text:
        candidates = sorted(RECORDINGS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"no recordings in {RECORDINGS_DIR}")
        return candidates[-1]
    path = Path(text)
    if not path.is_absolute():
        path = RECORDINGS_DIR / path
    if not path.suffix:
        path = path.with_suffix(".jsonl")
    if not path.is_file():
        raise FileNotFoundError(f"recording not found: {path}")
    return path


def recording_path_for(name: str | None) -> Path | None:
    """`<name>_<timestamp>.jsonl` under RECORDINGS_DIR, or None for the default unity_<timestamp> name."""
    safe = re.sub(r"[^\w\-]+", "_", (name or "").strip()).strip("_")[:60]
    if not safe:
        return None
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDINGS_DIR / f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"


def _upright_payload(upright):
    if upright is None:
        return None
    quats, accel = upright
    return {"quats": np.asarray(quats).tolist(), "accel": np.asarray(accel).tolist()}


class InputRecorder:
    def __init__(self, layout: str, labels: List[str], label_to_key: Dict[str, str],
                 acceleration_source: str, upright, model_fps: int, path: Path | None = None,
                 metadata: Dict | None = None) -> None:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = path or RECORDINGS_DIR / time.strftime("unity_%Y%m%d_%H%M%S.jsonl")
        self.labels = list(labels)
        self.label_to_key = dict(label_to_key)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._start = time.perf_counter()
        self.ticks = 0
        self._write({
            "type": "header", "format": FORMAT, "formatVersion": FORMAT_VERSION,
            "createdUnixSeconds": time.time(), "layout": layout, "labels": self.labels,
            "labelToTrackerKey": self.label_to_key, "accelerationSource": acceleration_source,
            "modelFps": model_fps, "upright": _upright_payload(upright), **(metadata or {}),
        })

    def _write(self, row) -> None:
        self._file.write(json.dumps(row, separators=(",", ":")) + "\n")

    def write_tick(self, latest: Dict[str, TrackerSample], ticks: int, calibration=None) -> None:
        self._write({
            "type": "tick", "t": time.perf_counter() - self._start, "ticks": ticks,
            "samples": {label: dataclasses.asdict(latest[self.label_to_key[label]]) for label in self.labels},
            "calibration": calibration,
        })
        self.ticks += ticks

    def write_coach(self, quats) -> None:
        """Coach pose from Unity: 24 SMPL-convention global quaternions (Unity world, xyzw)."""
        self._write({"type": "coach", "t": time.perf_counter() - self._start, "q": quats})

    def write_prediction(self, raw_quats, final_quats, diagnostics, calibration) -> None:
        """Actual model output is separate from coach animation in the same log."""
        self._write({"type": "prediction", "t": time.perf_counter() - self._start,
                     "rawLocalQuaternions": np.asarray(raw_quats).reshape(-1).tolist(),
                     "unityLocalQuaternions": np.asarray(final_quats).reshape(-1).tolist(),
                     "diagnostics": diagnostics, "calibration": calibration})

    def write_event(self, name: str, **payload) -> None:
        self._write({"type": "event", "name": name, "t": time.perf_counter() - self._start, **payload})

    @property
    def seconds(self) -> float:
        return self.ticks / 60.0

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


@dataclasses.dataclass
class InputRecording:
    path: Path
    layout: str
    labels: List[str]
    label_to_key: Dict[str, str]
    acceleration_source: str
    upright: Optional[tuple]
    times: List[float]
    samples: List[Dict[str, dict]]
    ticks: List[int]
    calibration: List[dict | None] = dataclasses.field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(max(1, count) for count in self.ticks) / 60.0


def load_input_recording(path: Path) -> InputRecording:
    with Path(path).open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows or rows[0].get("format") != FORMAT or rows[0].get("formatVersion") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a {FORMAT} v{FORMAT_VERSION} recording")
    header = rows[0]
    ticks = [row for row in rows[1:] if row.get("type") == "tick"]
    if not ticks:
        raise ValueError(f"{path} contains no input frames")
    upright = header.get("upright")
    return InputRecording(
        path=Path(path), layout=header["layout"], labels=list(header["labels"]),
        label_to_key=dict(header["labelToTrackerKey"]),
        acceleration_source=header.get("accelerationSource", "server"),
        upright=None if upright is None else (np.asarray(upright["quats"], dtype=np.float32),
                                              np.asarray(upright["accel"], dtype=np.float32)),
        times=[float(row["t"]) for row in ticks],
        samples=[row["samples"] for row in ticks],
        ticks=[int(row.get("ticks", 1)) for row in ticks],
        calibration=[row.get("calibration") for row in ticks],
    )


def _sample_for(raw: dict, tracker_key: str) -> TrackerSample:
    values = {field.name: raw.get(field.name) for field in dataclasses.fields(TrackerSample)}
    values["tracker_key"] = tracker_key
    for name, value in values.items():
        if isinstance(value, list):
            values[name] = tuple(value)
    return TrackerSample(**values)


class InputMux:
    """Single tracker-update stream that is fed by SolarXR or by a looping replay.

    Live updates are dropped while a replay is active, so a replay works with
    SlimeVR offline and an online SlimeVR cannot fight the recorded input.
    """

    def __init__(self, live_source, capacity: int = 16) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=capacity)
        self._replay_task: asyncio.Task | None = None
        self.replay: InputRecording | None = None
        self._live_task = asyncio.create_task(self._pump_live(live_source))

    @property
    def replaying(self) -> bool:
        return self._replay_task is not None

    def _offer(self, update) -> None:
        update = TimedTrackerUpdate(update, time.perf_counter())
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(update)

    def _clear(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    async def _pump_live(self, source) -> None:
        async for update in source:
            if self._replay_task is None:
                self._offer(update)

    async def _pump_replay(self, recording: InputRecording, label_to_key: Dict[str, str]) -> None:
        # Keyed by the *current* layout's tracker keys so `latest` lookups stay valid.
        frames = [{label_to_key[label]: _sample_for(raw[label], label_to_key[label]) for label in recording.labels}
                  for raw in recording.samples]
        period = recording.duration
        loop_start = time.perf_counter()
        while True:
            frame_index = 0
            for row_index, (update, count) in enumerate(zip(frames, recording.ticks)):
                for _ in range(max(1, count)):
                    delay = loop_start + frame_index / 60.0 - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # Replay is a verification source: preserve every recorded model
                    # frame even if inference briefly falls behind.
                    state = recording.calibration[row_index] if recording.calibration else None
                    await self._queue.put(TimedTrackerUpdate(
                        update, loop_start + frame_index / 60.0, calibration=state))
                    frame_index += 1
            loop_start += period

    def start_replay(self, recording: InputRecording, label_to_key: Dict[str, str]) -> None:
        self.stop_replay()
        self.replay = recording
        self._replay_task = asyncio.create_task(self._pump_replay(recording, label_to_key))

    def stop_replay(self) -> None:
        if self._replay_task is not None:
            self._replay_task.cancel()
        self._replay_task = None
        self.replay = None
        self._clear()

    def drain_pending(self) -> list:
        """All updates already queued, oldest first, without waiting."""
        pending = []
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return pending

    def wake(self) -> None:
        """Yield an empty update so the consumer can act on commands while no input flows."""
        self._offer({})

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._queue.get()


class TimedTrackerUpdate(dict):
    def __init__(self, update, received_monotonic: float, calibration=None):
        super().__init__(update)
        self.received_monotonic = received_monotonic
        self.calibration = calibration
