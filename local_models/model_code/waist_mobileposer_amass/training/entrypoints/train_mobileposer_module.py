#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, choices=["poser", "joints", "foot_contact", "velocity"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    from lightning.pytorch import seed_everything
    from mobileposer.config import train_hypers
    from mobileposer.constants import MODULES
    from mobileposer.train import TrainingManager

    seed_everything(args.seed, workers=True)
    train_hypers.device = args.gpu
    train_hypers.num_epochs = args.epochs
    args.output.mkdir(parents=True, exist_ok=True)
    manager = TrainingManager(fast_dev_run=args.fast_dev_run)
    manager.train_module(MODULES[args.module](), args.module, args.output)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
