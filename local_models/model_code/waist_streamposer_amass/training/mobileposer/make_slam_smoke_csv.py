"""Create a deterministic SLAM CSV from one processed sequence for smoke tests."""

import argparse
import csv
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--noise-m", type=float, default=0.03)
    args = parser.parse_args()

    blob = torch.load(args.sequence_file, map_location="cpu", weights_only=False)
    position = blob["tran"][args.sequence_index].float()
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(position.shape, generator=generator) * args.noise_m
    position = position - position[:1] + noise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz", "tracking_state"])
        for index, value in enumerate(position):
            writer.writerow([index / args.fps, *value.tolist(), 1.0, 0.0, 0.0, 0.0, 1])


if __name__ == "__main__":
    main()
