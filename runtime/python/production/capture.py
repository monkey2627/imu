"""Collect a reproducible SolarXR research session without running inference."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path

from runtime_contract import REALTIME_LAYOUT
from production.layout import LayoutMismatchError, detect_layout
from production.solarxr_client import stream_trackers

PHASES = (
    ("D0_stand", 1.0),
    ("D1_shoulder_flexion", 2.0),
    ("D2_shoulder_abduction", 2.0),
    ("D3_T_pose_elbow_flexion", 2.0),
    ("D4_fast_squat", 2.0),
    ("D5_leg_raise_kick", 2.0),
    ("D6_fast_step", 2.0),
)


def _json(value):
    if dataclasses.is_dataclass(value):
        return {key: _json(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


class SessionWriter:
    def __init__(self, root: Path, args):
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.imu = (root / "imu_solarxr.jsonl").open("w", encoding="utf-8", buffering=1)
        self.events = (root / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        metadata = {
            "format": "mobileposer-research-session",
            "formatVersion": 1,
            "createdUnixSeconds": time.time(),
            "timestampSource": "python_host_receive_clock; SolarXR hardware timestamp unavailable",
            "quaternionOrder": "xyzw",
            "accelerationUnits": "m/s2",
            "gyroUnits": "rad/s",
            "layoutRequested": args.layout,
            "videoPath": args.video_path,
            "referencePath": args.reference_path,
            "subjectId": args.subject_id,
            "notes": args.notes,
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def event(self, name, **payload):
        row = {
            "name": name,
            "hostUnixSeconds": time.time(),
            "hostMonotonicSeconds": time.perf_counter(),
            **_json(payload),
        }
        self.events.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def update(self, index, samples):
        row = {
            "sourceUpdate": index,
            "receiveUnixSeconds": time.time(),
            "receiveMonotonicSeconds": time.perf_counter(),
            "trackers": _json(samples),
        }
        self.imu.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self):
        self.imu.close()
        self.events.close()


async def run_capture(args):
    root = Path(args.output) / f"{args.subject_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SessionWriter(root, args)
    source = stream_trackers(minimum_time_since_last_ms=args.interval_ms).__aiter__()
    latest, count = {}, [0]

    async def next_update():
        update = await source.__anext__()
        latest.update(update)
        count[0] += 1
        writer.update(count[0], update)
        return update

    try:
        writer.event("session_started", phases=[phase[0] for phase in PHASES], validationSeconds=args.validation_seconds)
        while True:
            await next_update()
            try:
                layout = detect_layout(latest)
                break
            except LayoutMismatchError:
                continue
        if args.layout != "auto" and layout.name != args.layout:
            raise RuntimeError(
                f"Requested layout '{args.layout}', but SolarXR resolved '{layout.name}'. "
                "Fix tracker body-part bindings or pass --layout auto."
            )
        writer.event("layout_resolved", layout=layout.name, labels=layout.labels, trackerKeys=layout.label_to_tracker_key)
        print(f"Resolved {layout.name}: {layout.labels}", file=sys.stderr)
        for name, seconds in PHASES:
            writer.event("phase_started", phase=name, durationSeconds=seconds)
            print(f"[{name}] {seconds:.0f}s", file=sys.stderr)
            deadline = time.perf_counter() + seconds
            while time.perf_counter() < deadline:
                await next_update()
            writer.event("phase_ended", phase=name)
        writer.event("validation_started", durationSeconds=args.validation_seconds)
        print(f"[validation] {args.validation_seconds:.0f}s", file=sys.stderr)
        deadline = time.perf_counter() + args.validation_seconds
        while time.perf_counter() < deadline:
            await next_update()
        writer.event("validation_ended")
        writer.event("session_ended", totalUpdates=count[0])
        print(f"Saved {root} ({count[0]} SolarXR updates)", file=sys.stderr)
    except BaseException as exc:
        writer.event("session_aborted", error=f"{type(exc).__name__}: {exc}", totalUpdates=count[0])
        raise
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("recordings/research"))
    parser.add_argument("--subject-id", default="pilot")
    parser.add_argument("--layout", default=REALTIME_LAYOUT)
    parser.add_argument("--video-path", default="")
    parser.add_argument("--reference-path", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--validation-seconds", type=float, default=30.0)
    parser.add_argument("--interval-ms", type=int, default=16)
    try:
        asyncio.run(run_capture(parser.parse_args()))
    except KeyboardInterrupt:
        print("Capture interrupted; partial session remains on disk.", file=sys.stderr)


if __name__ == "__main__":
    main()
