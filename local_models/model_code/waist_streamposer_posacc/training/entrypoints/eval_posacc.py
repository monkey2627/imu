#!/usr/bin/env python3
"""Evaluation of PosAccPoser against the baselines and its controls, with criteria fixed BEFORE the runs
finished (2026-11-06).  Success means: telling a raised straight leg from a kneeling one, from the IMU.

Models (all streamed from a zero / 'unknown start' state, as live):
  stream_s42_amass             baseline (AMASS only)
  deployed                     stream_s42_ours_offsetfree_aug (Ours real-IMU fine-tune, waist offset as live)
  posacc_s42_amass             new model
  posacc_ctrl_stream_s42       control A: baseline architecture, same new data + noise
  posacc_s42_leave8            control B: new model, coach bird dog left out of training
Each with the real acceleration and with the acceleration set to zero (does the result come from it?).
Tests:
  1 coach, clean synthetic IMU from the labels (in-sample for models that saw Ours synth, except control B
    on bird dog): bird dog raised-leg AND kneeling frames, dead bug extended AND table-top frames;
  2 user, bird-dog calibration (calibration_20261103_215533): kneeling base, raised legs, supporting legs;
  3 user, negative test (unity_20261103_225546, always kneeling): still, shank wiggle, kicks.
Leg state = thigh elevation (deg; kneeling about -60..-90, raised straight about -30..0) / knee flexion.

  python eval_posacc.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import eval_thigh_vs_shank as ev                      # noqa: E402  (predict, metrics, body model)
from posacc_model import PosAccPoser                  # noqa: E402
from build_posacc_ours_synth import CFG, LABELS, synth_acc   # noqa: E402

OURS_LABEL_WAIST_OFFSET = torch.tensor([[0.98584, -0.00047, -0.16767], [0.05032, 0.95474, 0.29315],
                                        [0.15994, -0.29744, 0.94125]])
LEGS = {"L": (1, 4, 7), "R": (2, 5, 8)}


def load(name, dev):
    folder = ROOT / "checkpoints" / name
    cfg = json.loads((folder / "config.json").read_text())
    if cfg.get("arch") == "posacc":
        m = PosAccPoser(bool(cfg.get("gyro", 1)), int(cfg["hidden"]), int(cfg["layers"]), frame="both")
        m.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
        return m.to(dev).eval(), int(cfg["lag"])
    weights = "model_finetuned.pt" if (folder / "model_finetuned.pt").exists() else "model.pt"
    return ev.load_model(folder, dev, weights)


MODELS = {"baseline stream_s42_amass": "stream_s42_amass", "deployed (Ours fine-tune)": "stream_s42_ours_offsetfree_aug",
          "NEW posacc_s42_amass": "posacc_s42_amass", "ctrl A: stream + new data": "posacc_ctrl_stream_s42",
          "ctrl B: posacc, no bird dog": "posacc_s42_leave8"}


def states(body, pred):
    _, j = body.forward_kinematics(pred)[:2]
    j = j.cpu().numpy()
    out = {}
    for side, (hip, knee, ankle) in LEGS.items():
        t, s = j[:, knee] - j[:, hip], j[:, ankle] - j[:, knee]
        t /= np.linalg.norm(t, axis=-1, keepdims=True); s /= np.linalg.norm(s, axis=-1, keepdims=True)
        out[side] = np.stack([np.degrees(np.arcsin(t[:, 1])), np.degrees(np.arccos(np.clip((t * s).sum(-1), -1, 1)))], 1)
    return out


@torch.no_grad()
def coach_synth(cap, body, dev):
    tgt = torch.load(ev.DATA / "captures" / cap / "targets_humanpose_calibrated_60.pt", map_location="cpu", weights_only=False)
    pose, tran = tgt["pose"].float().to(dev), tgt["tran"].float().to(dev)
    glb, _, verts = body.forward_kinematics(pose, tran=tran, calc_mesh=True)
    pts = [(verts[:, CFG["attachments"][l]["vertices"]] * torch.tensor(CFG["attachments"][l]["weights"], device=dev)[None, :, None]).sum(1)
           for l in LABELS]
    acc = synth_acc(torch.stack(pts, 1)).cpu()
    ori = glb[:, [CFG["attachments"][l]["joint"] for l in LABELS]].cpu()
    return acc, ori, pose


def run(model, lag, body, acc, ori, dev, deployed):
    ori = ori.clone()
    if deployed:
        ori[:, 4] = ori[:, 4] @ OURS_LABEL_WAIST_OFFSET
    return ev.predict(model, lag, body, acc, ori, dev)


def mean_state(st, side, a, b):
    return st[side][a:b].mean(0)


def main():
    dev = torch.device("cuda:0")
    body = ev.ParametricModel(ev.paths.smpl_file, device=dev)
    models = {k: load(v, dev) for k, v in MODELS.items()}
    results = {}
    coach = {c: coach_synth(c, body, dev) for c in ("8", "7")}
    user = np.load(ROOT / "artifacts/user_birddog_215533.npz")
    neg = np.load(ROOT / "artifacts/user_negative_225546.npz")
    B, L, R = int(user["base"]), int(user["left_reach"]), int(user["right_reach"])
    for mname, (model, lag) in models.items():
        deployed = mname.startswith("deployed")
        for acc_on in (True, False):
            tag = f"{mname}{'' if acc_on else ' | acc=0'}"
            row = {}
            # 1 coach synthetic: both states
            for cap, (acc, ori, pose) in coach.items():
                a = acc if acc_on else torch.zeros_like(acc)
                pred = run(model, lag, body, a, ori, dev, deployed)
                rows = ev.leg_metrics(body, pred, pose[:len(pred)])
                if cap == "8":
                    row["coach BD raised: knee err"] = ev.summarise(rows, lambda r: r["label_thigh_el"] > -45)["knee_flex_err"]
                    row["coach BD kneeling: knee err"] = ev.summarise(rows, lambda r: (r["label_thigh_el"] < -60) & (r["label_knee_flex"] > 60))["knee_flex_err"]
                else:
                    row["coach DB extended: knee err"] = ev.summarise(rows, lambda r: (r["label_knee_flex"] < 45) & (r["label_thigh_el"] < 60))["knee_flex_err"]
                    row["coach DB table-top: knee err"] = ev.summarise(rows, lambda r: (r["label_knee_flex"] > 70) & (r["label_thigh_el"] > 60))["knee_flex_err"]
            # 2 user bird dog
            acc, ori = torch.from_numpy(user["acc"]), torch.from_numpy(user["ori"])
            st = states(body, run(model, lag, body, acc if acc_on else torch.zeros_like(acc), ori, dev, deployed))
            row["user base (kneel) L,R"] = [mean_state(st, s, B - 180, B).round(0).tolist() for s in "LR"]
            row["user raised R / L"] = [mean_state(st, "R", L - 180, L).round(0).tolist(), mean_state(st, "L", R - 180, R - 1).round(0).tolist()]
            row["user supporting L / R"] = [mean_state(st, "L", L - 180, L).round(0).tolist(), mean_state(st, "R", R - 180, R - 1).round(0).tolist()]
            # 3 user negative (always kneeling)
            acc, ori = torch.from_numpy(neg["acc"]), torch.from_numpy(neg["ori"])
            st = states(body, run(model, lag, body, acc if acc_on else torch.zeros_like(acc), ori, dev, deployed))
            for seg, (a, b) in (("still", (0, 270)), ("R wiggle", (270, 540)), ("kicks", (960, 1200))):
                row[f"neg {seg} (kneel) L,R"] = [mean_state(st, s, a, b).round(0).tolist() for s in "LR"]
            results[tag] = row
            print(f"\n{tag}")
            for k, v in row.items():
                print(f"   {k:32s} {v if not isinstance(v, float) else round(v, 1)}", flush=True)
    (ROOT / "artifacts/eval_posacc.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
