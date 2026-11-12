#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--crop", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from mobileposer.stream.data import Store
    from mobileposer.stream.model import StreamPoser
    from mobileposer.stream.train import lagged_loss

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    cfg = json.loads((args.init / "config.json").read_text())
    model = StreamPoser(
        bool(cfg["gyro"]), cfg["hidden"], cfg["layers"],
        anchor=cfg.get("anchor", "none"), frame=cfg.get("frame", "rel"),
        arch=cfg.get("arch", "mp2"), local=bool(cfg.get("local_gyro", 0)),
    ).to(dev)
    model.load_state_dict(torch.load(args.init / "model.pt", map_location=dev))
    data = Store(args.store, dev)
    loss_args = SimpleNamespace(
        gyro=cfg["gyro"], lag=cfg["lag"], wj=cfg["wj"],
    )
    with torch.random.fork_rng(devices=[dev]):
        torch.manual_seed(0)
        checks = [data.sample(args.bs, args.crop) for _ in range(8)]

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            return sum(float(lagged_loss(model, *b, loss_args, 0.0)[0]) for b in checks) / len(checks)

    before = evaluate()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for step in range(1, args.iters + 1):
        model.train()
        batch = data.sample(args.bs, args.crop)
        loss, rot, joint = lagged_loss(model, *batch, loss_args, 0.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == 1:
            row = {"step": step, "loss": float(loss), "rot": rot, "joint": joint}
            history.append(row)
            print(json.dumps(row), flush=True)
    after = evaluate()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "model.pt")
    result = {
        "kind": "same-sequence deadbug overfit",
        "seed": args.seed,
        "iterations": args.iters,
        "learning_rate": args.lr,
        "before_loss": before,
        "after_loss": after,
        "train_and_evaluate_same_sequence": True,
        "input": "five physical IMUs: lw,rw,ls,rs,waist",
        "supervision": "same-capture FBX-to-SMPL pseudo-labels",
        "pretrained": str(args.init.resolve()),
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps({**cfg, "personalization": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
