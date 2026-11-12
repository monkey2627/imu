#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def best(folder: Path) -> Path:
    files = list(folder.glob("*.ckpt"))
    if not files:
        raise FileNotFoundError(folder)
    return min(files, key=lambda p: float(p.name.split("validation_step_loss=")[1].split(".ckpt")[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mobileposer.constants import MODULES
    from mobileposer.models import MobilePoserNet

    loaded = {}
    manifest = {}
    for name, cls in MODULES.items():
        checkpoint = best(args.root / name)
        loaded[name] = cls.load_from_checkpoint(checkpoint, map_location="cpu")
        manifest[name] = {"checkpoint": str(checkpoint.resolve()), "sha256": digest(checkpoint)}
    model = MobilePoserNet(
        poser=loaded["poser"],
        joints=loaded["joints"],
        foot_contact=loaded["foot_contact"],
        velocity=loaded["velocity"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    manifest["combined"] = {"path": str(args.output.resolve()), "sha256": digest(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
