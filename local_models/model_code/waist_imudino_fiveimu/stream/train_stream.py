"""Train the stateful five-IMU head on streaming token caches.

Crops of CROP frames start at random stream positions; the GRU starts from zero
state at the crop start (as online does after start-up) and the first WARMUP
frames are excluded from the loss. Loss = 6D MSE + VEL_WEIGHT * MSE of the
frame-to-frame 6D difference against the target's (matches motion, does not
damp it). Validation is per source group and also reports the 2-frame jitter
of predictions vs. ground truth on full held-out sequences.

    python -m stream.train_stream --cache-root data/features/stream_wristankle_v1 --output runs/stream_wristankle
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch import nn

from downstream.prepare import AMASS_NAMES
from .common import TOKEN_DIM, frame_features, token_slot
from .model import StreamHead


def sixd_to_matrix(values):
    first, second = values[..., :3], values[..., 3:6]
    c1 = nn.functional.normalize(first, dim=-1, eps=1e-8)
    second = second - (c1 * second).sum(-1, keepdim=True) * c1
    c2 = nn.functional.normalize(second, dim=-1, eps=1e-8)
    return torch.stack((c1, c2, torch.cross(c1, c2, dim=-1)), dim=-1)


def angle_deg(a, b):
    tr = (a * b).sum((-1, -2))
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


class Sequences:
    def __init__(self, paths, val):
        self.items = []
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=False)
            bird = path.name.startswith("BirdDog")
            for s in data["sequences"]:
                if s["val"] == val:
                    frames = torch.arange(len(s["imu"]))
                    s["frame_feat"] = frame_features(s["imu"], s["ori"], frames)
                    s["slot"] = token_slot(frames)
                    s["bird"] = bird
                    del s["imu"], s["ori"]
                    self.items.append(s)
        self.lengths = torch.tensor([len(s["target"]) for s in self.items], dtype=torch.float)

    def gather(self, s, lo, hi):
        slot = s["slot"][lo:hi]
        tok = torch.zeros((hi - lo, TOKEN_DIM))
        ok = slot >= 0
        tok[ok] = s["tokens"][slot[ok]]
        return tok, s["frame_feat"][lo:hi], s["target"][lo:hi].float()

    def batch(self, size, crop, bird_fraction):
        toks, feats, tgts, birds = [], [], [], []
        bird_ids = [i for i, s in enumerate(self.items) if s["bird"]]
        for _ in range(size):
            if bird_ids and random.random() < bird_fraction:
                i = random.choice(bird_ids)
            else:
                i = int(torch.multinomial(self.lengths, 1))
            s = self.items[i]
            n = len(s["target"])
            length = min(crop, n)
            lo = random.randint(0, n - length)
            tok, feat, tgt = self.gather(s, lo, lo + length)
            pad = crop - length
            if pad:
                tok = torch.cat((tok, tok.new_zeros(pad, TOKEN_DIM)))
                feat = torch.cat((feat, feat.new_zeros(pad, feat.shape[1])))
                tgt = torch.cat((tgt, tgt.new_full((pad, tgt.shape[1]), float("nan"))))
            toks.append(tok), feats.append(feat), tgts.append(tgt), birds.append(s["bird"])
        return torch.stack(toks), torch.stack(feats), torch.stack(tgts), torch.tensor(birds)


def loss_fn(pred, tgt, warmup, vel_weight):
    pred, tgt = pred[:, warmup:], tgt[:, warmup:]
    valid = torch.isfinite(tgt[..., 0])
    t = torch.nan_to_num(tgt)
    pos = ((pred - t) ** 2).mean(-1)[valid].mean()
    v_valid = valid[:, 1:] & valid[:, :-1]
    vel = (((pred[:, 1:] - pred[:, :-1]) - (t[:, 1:] - t[:, :-1])) ** 2).mean(-1)[v_valid].mean()
    return pos + vel_weight * vel, pos, vel


@torch.no_grad()
def validate(head, val, device, warmup, max_seqs=400):
    head.eval()
    out = {"amass": [], "birddog": []}
    jit = {"amass": [], "birddog": []}
    gt_jit = {"amass": [], "birddog": []}
    items = val.items[:: max(1, len(val.items) // max_seqs)]
    for s in items:
        n = len(s["target"])
        tok, feat, tgt = val.gather(s, 0, n)
        pred, _ = head(tok[None].to(device), feat[None].to(device))
        p = sixd_to_matrix(pred[0, warmup:].float().cpu().reshape(-1, 23, 6))
        g = sixd_to_matrix(tgt[warmup:].reshape(-1, 23, 6))
        if len(p) < 3:
            continue
        key = "birddog" if s["bird"] else "amass"
        out[key].append(angle_deg(p, g).mean())
        jit[key].append(angle_deg(p[2:], p[:-2]).flatten())
        gt_jit[key].append(angle_deg(g[2:], g[:-2]).flatten())
    report = {}
    for key in out:
        if out[key]:
            j, gj = torch.cat(jit[key]), torch.cat(gt_jit[key])
            report[key] = {"meanLocalAngleDeg": round(float(torch.stack(out[key]).mean()), 3),
                           "pred2frame_p99": round(float(torch.quantile(j[:2_000_000], .99)), 2),
                           "gt2frame_p99": round(float(torch.quantile(gj[:2_000_000], .99)), 2)}
    head.train()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=[*AMASS_NAMES, "BirdDog"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--crop", type=int, default=360)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vel-weight", type=float, default=1.0)
    ap.add_argument("--bird-fraction", type=float, default=0.15)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    paths = [args.cache_root / f"{d}.stream.pt" for d in args.datasets]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    train, val = Sequences(paths, False), Sequences(paths, True)
    leak = {s["group"] for s in train.items} & {s["group"] for s in val.items}
    if leak:
        raise ValueError(f"group leakage: {list(leak)[:3]}")
    print(f"[data] train={len(train.items)} val={len(val.items)} "
          f"train_frames={int(train.lengths.sum())}", flush=True)
    head = StreamHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)
    args.output.mkdir(parents=True, exist_ok=True)
    latest, best_path = args.output / "latest.pt", args.output / "best.pt"
    step, best = 0, math.inf
    if latest.exists():
        st = torch.load(latest, map_location="cpu", weights_only=False)
        head.load_state_dict(st["head"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best = st["step"], st["best"]
        print(f"[resume] step={step}", flush=True)
    running = []
    while step < args.steps:
        tok, feat, tgt, _ = train.batch(args.batch, args.crop, args.bird_fraction)
        pred, _ = head(tok.to(device, non_blocking=True), feat.to(device, non_blocking=True))
        loss, pos, vel = loss_fn(pred, tgt.to(device), args.warmup, args.vel_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        running.append((pos.item(), vel.item()))
        if step % 200 == 0:
            p, v = (sum(x) / len(running) for x in zip(*running))
            print(f"[train] step={step} pos={p:.5f} vel={v:.6f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
            running.clear()
        if step % args.val_every == 0 or step == args.steps:
            rep = validate(head, val, device, args.warmup)
            score = sum(r["meanLocalAngleDeg"] for r in rep.values()) / len(rep)
            state = {"head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "step": step, "best": min(best, score), "args": vars(args) | {"cache_root": str(args.cache_root),
                     "output": str(args.output)}, "contract": "stream.common v1",
                     "sensor_order": ["lw", "rw", "ls", "rs", "waist"], "target": "SMPL local 6D joints 1..23"}
            torch.save(state, latest.with_suffix(".tmp")); os.replace(latest.with_suffix(".tmp"), latest)
            if score < best:
                best = score
                torch.save(state, best_path.with_suffix(".tmp")); os.replace(best_path.with_suffix(".tmp"), best_path)
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "val": rep, "score": score, "best": best}) + "\n")
            print(f"[val] step={step} {json.dumps(rep)} best={best:.3f}", flush=True)
    (args.output / "COMPLETE").write_text(f"steps={args.steps}\n")


if __name__ == "__main__":
    main()
