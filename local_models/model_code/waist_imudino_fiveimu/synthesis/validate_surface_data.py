#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch


LABELS = ["lw", "rw", "ls", "rs", "waist"]
JOINTS = [18, 19, 4, 5, 3]
WAIST_FACE = 6840
WAIST_VERTICES = [1335, 3504, 1767]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    waist = config["attachments"]["waist"]
    assert waist["joint"] == 3
    assert waist["face"] == WAIST_FACE
    assert waist["vertices"] == WAIST_VERTICES
    assert abs(sum(waist["weights"]) - 1.0) < 1e-8
    assert max(waist["weights"]) < 0.999, "waist must be a barycentric surface point"

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    synthesis = data["synthesis"]
    assert data["layout"] == "wrists_shanks_waist"
    assert data["layout_labels"] == LABELS
    assert data["layout_joints"] == JOINTS
    assert synthesis["mode"] == "surface"
    assert synthesis["fps"] == 60
    cached_waist = synthesis["attachments"]["attachments"]["waist"]
    assert cached_waist == waist
    assert synthesis["attachment_sha256"] == sha256(args.config)
    assert all(x.shape[1:] == (6, 3) for x in data["acc"])
    assert all(x.shape[1:] == (6, 3, 3) for x in data["ori"])
    assert all(len(a) == len(o) == len(p) for a, o, p in zip(data["acc"], data["ori"], data["pose"]))

    report = {
        "status": "ok",
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "attachment_config": str(args.config.resolve()),
        "attachment_sha256": sha256(args.config),
        "layout": data["layout"],
        "labels": LABELS,
        "orientation_bones": JOINTS,
        "fps": 60,
        "sequence_count": len(data["pose"]),
        "frame_count": sum(map(len, data["pose"])),
        "waist_surface": {
            "face": waist["face"],
            "vertices": waist["vertices"],
            "weights": waist["weights"],
            "neutral_position_m": waist["neutral_position_m"],
            "description": waist["description"],
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
