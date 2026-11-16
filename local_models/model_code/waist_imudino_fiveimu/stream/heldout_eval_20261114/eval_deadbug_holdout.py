"""Dead-bug held-out evaluation of the stream head, with leg-specific metrics.

Same inference as stream/eval_stream.py and the trainer's validation: every sequence is decoded in
one causal pass from zero GRU state with the cached tokens; the first WARMUP frames are excluded.
Split = the cached ``val`` flag (group_fold of the source video, 3 videos / 56 clips held out).
The original DeadBug *training* clips are evaluated the same way for comparison (DeadBugMix is
recombined training data and is not evaluated).

Run (mobileposer env):
  PYTHONPATH=$BASE/synth_ws:$BASE/imudino_stream python eval_deadbug_holdout.py --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from stream.common import TOKEN_DIM, frame_features, token_slot
from stream.model import StreamHead

N = Path("/home/duanyuhan/dyh/motion/lines/imudino_deadbug_20261113")
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
WARMUP = 60
JOINT_NAMES = {1: "L_hip", 2: "R_hip", 4: "L_knee", 5: "R_knee", 16: "L_shoulder", 17: "R_shoulder",
               18: "L_elbow", 19: "R_elbow", 3: "spine1", 9: "spine3"}


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


def leg_angles(glob):
    """Thigh angle in the spine1 sagittal plane (0 = toward the feet, 90 = chest normal) and knee flexion."""
    spine = glob[:, 3]
    a_axis, b_axis = -spine[..., :, 1], spine[..., :, 2]
    down = torch.tensor([0.0, -1.0, 0.0])
    out = {}
    for side, hip, knee in (("L", 1, 4), ("R", 2, 5)):
        thigh, shank = glob[:, hip] @ down, glob[:, knee] @ down
        out[side + "_thigh"] = torch.rad2deg(torch.atan2((thigh * b_axis).sum(-1), (thigh * a_axis).sum(-1)))
        out[side + "_knee"] = torch.rad2deg(torch.arccos((thigh * shank).sum(-1).clamp(-1, 1)))
    return out


def state(thigh, knee):
    """0 = extended/lowered, 1 = tabletop/flexed, -1 = in between (excluded from the confusion)."""
    s = torch.full_like(thigh, -1, dtype=torch.long)
    s[(thigh < 45) & (knee < 40)] = 0
    s[(thigh > 60) & (knee > 50)] = 1
    return s


@torch.no_grad()
def run(head, seq, device):
    n = len(seq["target"])
    frames = torch.arange(n)
    slot = token_slot(frames)
    tok = torch.zeros((n, TOKEN_DIM))
    ok = slot >= 0
    tok[ok] = seq["tokens"][slot[ok]].float()
    feat = frame_features(seq["imu"], seq["ori"], frames)
    pred, _ = head(tok[None].to(device), feat[None].to(device))
    return sixd(pred[0].float().cpu().reshape(-1, 23, 6)), sixd(seq["target"].float().reshape(-1, 23, 6))


def evaluate(head, seqs, body, device, keep_predictions=False):
    rows, preds = [], []
    for index, s in seqs:
        p, t = run(head, s, device)
        waist = s["ori"][:, 4].float()
        gt_root = waist @ t[:, 2].transpose(-1, -2)
        dep_root = waist @ p[:, 2].transpose(-1, -2)
        tl = torch.cat((gt_root[:, None], t), 1)
        dl = torch.cat((dep_root[:, None], p), 1)
        gl = torch.cat((gt_root[:, None], p), 1)
        w = slice(WARMUP, None)
        g_t, g_d = fk_rot(tl), fk_rot(dl)
        _, j_t = body.forward_kinematics(tl[w])[:2]
        _, j_d = body.forward_kinematics(gl[w])[:2]
        local_err = angle(p[w], t[w])                       # (T,23) joints 1..23
        la_t, la_p = leg_angles(g_t[w]), leg_angles(g_d[w])
        row = {"cache_index": index, "group": s["group"], "frames": len(t) - WARMUP,
               "local": local_err.mean().item(),
               "deploy_ang": angle(g_d[w], g_t[w]).mean().item(),
               "deploy_sip": angle(g_d[w][:, [1, 2, 16, 17]], g_t[w][:, [1, 2, 16, 17]]).mean().item(),
               "pos_cm": ((j_d - j_d[:, :1]) - (j_t - j_t[:, :1])).norm(dim=-1).mean().item() * 100}
        for j, name in JOINT_NAMES.items():
            row["local_" + name] = local_err[:, j - 1].mean().item()
        for key in la_t:
            row[key + "_abs_err"] = (la_p[key] - la_t[key]).abs().mean().item()
            row[key + "_gt_range"] = [la_t[key].min().item(), la_t[key].max().item()]
        row["_angles"] = {k: (la_t[k].numpy(), la_p[k].numpy()) for k in la_t}
        rows.append(row)
        if keep_predictions:
            preds.append({"cache_index": index, "group": s["group"], "pred_local": dl.numpy().astype(np.float32),
                          "gt_local": tl.numpy().astype(np.float32), "waist": waist.numpy()})
    return rows, preds


def aggregate(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    out = {"sequences": len(rows), "frames": int(sum(r["frames"] for r in rows))}
    for k in keys:   # mean over frames, then over sequences (as eval_stream)
        out[k] = round(float(np.mean([r[k] for r in rows])), 2)
    conf = np.zeros((2, 2, 2), int)   # side, gt, pred
    thigh_gt, thigh_pred = {"L": [], "R": []}, {"L": [], "R": []}
    for r in rows:
        for si, side in enumerate("LR"):
            t_th, p_th = r["_angles"][side + "_thigh"]
            t_kn, p_kn = r["_angles"][side + "_knee"]
            gs, ps = state(torch.from_numpy(t_th), torch.from_numpy(t_kn)), state(torch.from_numpy(p_th), torch.from_numpy(p_kn))
            for g in (0, 1):
                for q in (0, 1):
                    conf[si, g, q] += int(((gs == g) & (ps == q)).sum())
            thigh_gt[side].append(t_th)
            thigh_pred[side].append(p_th)
    for si, side in enumerate("LR"):
        g, p = np.concatenate(thigh_gt[side]), np.concatenate(thigh_pred[side])
        out[f"{side}_thigh_corr"] = round(float(np.corrcoef(g, p)[0, 1]), 3)
        out[f"{side}_thigh_err_if_constant_mean"] = round(float(np.abs(g - g.mean()).mean()), 2)
        c = conf[si]
        out[f"{side}_state_confusion[gt][pred]"] = {"extended->extended": int(c[0, 0]), "extended->tabletop": int(c[0, 1]),
                                                    "tabletop->extended": int(c[1, 0]), "tabletop->tabletop": int(c[1, 1])}
        out[f"{side}_extended_recall"] = round(float(c[0, 0] / max(c[0].sum(), 1)), 3)
        out[f"{side}_tabletop_recall"] = round(float(c[1, 1] / max(c[1].sum(), 1)), 3)
        out[f"{side}_gt_extended_frames_share"] = round(float(c[0].sum() / max(c.sum(), 1)), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(N / "runs/stream_deadbug_limbmix/best.pt"))
    ap.add_argument("--cache", default=str(N / "data/features/stream_wristankle_v1/DeadBug.stream.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head = StreamHead().eval().to(device)
    head.load_state_dict(state_dict["head"], strict=True)
    seqs = torch.load(args.cache, map_location="cpu", weights_only=False)["sequences"]
    body = ParametricModel(paths.smpl_file, device="cpu")
    report = {"checkpoint": args.checkpoint, "checkpoint_step": state_dict.get("step"), "warmup_frames": WARMUP,
              "state_thresholds": "extended: thigh<45 & knee<40; tabletop: thigh>60 & knee>50 (deg)"}
    for name, val in (("heldout", True), ("train_original_clips", False)):
        chosen = [(i, s) for i, s in enumerate(seqs) if bool(s["val"]) == val]
        rows, preds = evaluate(head, chosen, body, device, keep_predictions=val)
        report[name] = aggregate(rows)
        report[name + "_per_video"] = {}
        for group in sorted({r["group"] for r in rows}):
            report[name + "_per_video"][group[-12:]] = aggregate([r for r in rows if r["group"] == group])
        if val:
            report["heldout_per_sequence"] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                                               if not k.startswith("_")} for r in rows]
            torch.save(preds, out / "heldout_predictions.pt")
        print(name, json.dumps(report[name]), flush=True)
    (out / "deadbug_holdout_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
