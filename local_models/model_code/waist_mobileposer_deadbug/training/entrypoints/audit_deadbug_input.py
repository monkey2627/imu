#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch


ROLES = {
    "lw": "LEFT_LOWER_ARM",
    "rw": "RIGHT_LOWER_ARM",
    "ls": "LEFT_LOWER_LEG",
    "rs": "RIGHT_LOWER_LEG",
    "waist": "WAIST",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    raw = args.capture / manifest["rawImuJsonFile"]
    assert sha256(raw) == manifest["rawImuJsonSha256"]
    package = json.loads(raw.read_text())
    frames = package["frames"]
    sensors = {}
    for label, role in ROLES.items():
        rows = []
        for frame in frames:
            matches = [x for x in frame["sensors"] if x["trackerRole"] == role]
            assert len(matches) == 1, (role, len(matches))
            rows.append(matches[0])
        ids = sorted({x["sensorId"] for x in rows})
        assert len(ids) == 1
        required = ("online", "hasAcceleration", "hasOrientation", "hasRotationReferenceAdjusted")
        assert all(all(x.get(key, False) for key in required) for x in rows)
        assert all(x["accelerationKind"] == "linear" for x in rows)
        sensors[label] = {"tracker_role": role, "sensor_id": ids[0], "frames": len(rows), "all_valid": True}
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    assert data["layout_labels"] == list(ROLES)
    assert data["fps"] == 60
    result = {
        "status": "ok",
        "capture": str(args.capture.resolve()),
        "raw_imu": str(raw.resolve()),
        "raw_imu_sha256": sha256(raw),
        "source_frames": len(frames),
        "source_duration_s": frames[-1]["timeSeconds"] - frames[0]["timeSeconds"],
        "output": str(args.input.resolve()),
        "output_sha256": sha256(args.input),
        "output_fps": data["fps"],
        "output_frames": len(data["acc"][0]),
        "layout": data["layout"],
        "labels": data["layout_labels"],
        "sensors": sensors,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
