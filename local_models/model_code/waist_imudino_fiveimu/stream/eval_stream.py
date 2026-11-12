"""Held-out evaluation of stream heads on the cached validation folds (AMASS sequences, BirdDog videos).

Every validation sequence is decoded in one causal pass from zero GRU state; the
first WARMUP frames are excluded. BirdDogMix is train-only and never evaluated.
Metrics (mean over frames, then over sequences within a domain):
  local   mean local-rotation error of SMPL joints 1..23 (deg)
  ang     global-rotation error over all 24 joints (deg)
  sip     global-rotation error of hips + shoulders (1, 2, 16, 17) (deg)
  pos     root-aligned joint-position error, 24 joints, neutral shape (cm)
  jitter  mean |third difference| of predicted joint positions (m/s^3); gtJitter for reference
Two root policies: "gtroot" uses the ground-truth pelvis; "deploy" uses the live rule
R_root = R_waist_sensor @ R_spine1_pred^T with the synthetic waist sensor.

Run with the mobileposer env from the synthesis workspace (needs mobileposer.articulate):
    PYTHONPATH=<synth_ws>:<imudino_stream> python -m stream.eval_stream --cache-root ... --checkpoints a.pt b.pt
"""
import argparse
import json
from pathlib import Path

import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths

from .common import TOKEN_DIM, frame_features, token_slot
from .model import StreamHead

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SIP = [1, 2, 16, 17]
WARMUP = 60
FPS = 60


def sixd(values):
    a, b = values[..., :3], values[..., 3:6]
    c1 = torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
    b = b - (c1 * b).sum(-1, keepdim=True) * c1
    c2 = torch.nn.functional.normalize(b, dim=-1, eps=1e-8)
    return torch.stack((c1, c2, torch.cross(c1, c2, dim=-1)), dim=-1)


def angle(a, b):
    return torch.rad2deg(torch.arccos((((a * b).sum((-1, -2)) - 1) / 2).clamp(-1, 1)))


def fk_rot(local):
    g = [None] * 24
    for j, p in enumerate(PARENTS):
        g[j] = local[:, j] if p < 0 else g[p] @ local[:, j]
    return torch.stack(g, 1)


@torch.no_grad()
def evaluate(head, seqs, body, device):
    per = {}
    for s in seqs:
        n = len(s["target"])
        frames = torch.arange(n)
        slot = token_slot(frames)
        tok = torch.zeros((n, TOKEN_DIM))
        ok = slot >= 0
        tok[ok] = s["tokens"][slot[ok]].float()
        feat = frame_features(s["imu"], s["ori"], frames)
        pred, _ = head(tok[None].to(device), feat[None].to(device))
        p = sixd(pred[0, WARMUP:].float().cpu().reshape(-1, 23, 6))
        t = sixd(s["target"][WARMUP:].float().reshape(-1, 23, 6))
        waist = s["ori"][WARMUP:, 4].float()
        gt_root = waist @ t[:, 2].transpose(-1, -2)          # synthetic waist sensor == GT spine1 global
        pl = torch.cat((gt_root[:, None], p), 1)
        tl = torch.cat((gt_root[:, None], t), 1)
        dl = torch.cat(((waist @ p[:, 2].transpose(-1, -2))[:, None], p), 1)
        gt_g = fk_rot(tl)
        _, gt_j = body.forward_kinematics(tl)[:2]
        gt_j = gt_j - gt_j[:, :1]
        rows = {"local": angle(p, t).mean(1)}
        for name, local in (("gtroot", pl), ("deploy", dl)):
            g = fk_rot(local)
            err = angle(g, gt_g)
            _, j = body.forward_kinematics(local)[:2]
            j = j - j[:, :1]
            rows[f"{name}_ang"] = err.mean(1)
            rows[f"{name}_sip"] = err[:, SIP].mean(1)
            rows[f"{name}_pos"] = (j - gt_j).norm(dim=-1).mean(1) * 100
            if name == "gtroot":
                jerk = (j[3:] - 3 * j[2:-1] + 3 * j[1:-2] - j[:-3]).norm(dim=-1).mean(1) * FPS ** 3
                gjerk = (gt_j[3:] - 3 * gt_j[2:-1] + 3 * gt_j[1:-2] - gt_j[:-3]).norm(dim=-1).mean(1) * FPS ** 3
                rows["jitter"], rows["gtJitter"] = jerk, gjerk
        for k, v in rows.items():
            per.setdefault(s["domain"], {}).setdefault(k, []).append(float(v.mean()))
    out = {}
    for dom, metrics in per.items():
        out[dom] = {"sequences": len(next(iter(metrics.values())))} | {
            k: round(sum(v) / len(v), 3) for k, v in metrics.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    seqs = []
    for path in sorted(args.cache_root.glob("*.stream.pt")):
        if path.name.startswith("BirdDogMix"):
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        dom = "birddog" if path.name.startswith("BirdDog") else "amass"
        for s in data["sequences"]:
            if s["val"] and len(s["target"]) > WARMUP + 8:
                s["domain"] = dom
                seqs.append(s)
    body = ParametricModel(paths.smpl_file, device="cpu")
    device = torch.device(args.device)
    report = {}
    for ck in args.checkpoints:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        head = StreamHead().to(device).eval()
        head.load_state_dict(state["head"])
        report[str(ck)] = {"step": state["step"], **evaluate(head, seqs, body, device)}
        print(json.dumps({str(ck): report[str(ck)]}), flush=True)


if __name__ == "__main__":
    main()
