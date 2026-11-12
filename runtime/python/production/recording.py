"""Lossless JSONL diagnostics for the SolarXR -> model -> Unity production path."""
from __future__ import annotations

import dataclasses
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

FORMAT = "mobileposer-realtime-jsonl"
FORMAT_VERSION = 1


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


class RealtimeRecorder:
    def __init__(self, path: Path, metadata: Dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self.write("metadata", {
            "format": FORMAT,
            "formatVersion": FORMAT_VERSION,
            "createdUnixSeconds": time.time(),
            "timestampSource": "python_host_receive_and_process_time_not_sensor_time",
            "quaternionOrder": "xyzw",
            "accelerationUnits": "m/s2",
            "accelerationKind": "gravity_removed_linear",
            "python": sys.version,
            "platform": platform.platform(),
            **metadata,
        })

    def write(self, record_type: str, payload: Dict[str, Any]) -> None:
        self._file.write(json.dumps(
            {"type": record_type, **_json_value(payload)}, ensure_ascii=False,
            separators=(",", ":")) + "\n")

    def event(self, name: str, **payload: Any) -> None:
        self.write("event", {
            "name": name,
            "hostUnixSeconds": time.time(),
            "hostMonotonicSeconds": time.perf_counter(),
            **payload,
        })

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc


def load_recording(path: Path) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    rows = list(read_jsonl(path))
    if not rows or rows[0].get("type") != "metadata":
        raise ValueError(f"{path} does not start with a metadata record")
    metadata = rows[0]
    if metadata.get("format") != FORMAT or metadata.get("formatVersion") != FORMAT_VERSION:
        raise ValueError(f"unsupported recording format: {metadata.get('format')!r} v{metadata.get('formatVersion')!r}")
    return metadata, rows[1:]
