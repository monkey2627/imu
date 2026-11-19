"""Lying-cat contact rule on a real recording: limb geometry, contact error, side views.

python -m analysis.lyingcat_rule_eval [recording] [--model ID] [--tag NAME]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from analysis.realuse_action_eval import _angle, metrics
from production.arm_prior import PARENTS, _globals
from production.deadbug_replay import replay
from production.lyingcat_contact import MAX_SPINE_DEG, LyingCatContact
from production.model_backends import ACTION_RUNTIME_MODELS
from production.paths import paths

UP = np.array([0.0, 1.0, 0.0])


def _bone(g, joint, child, offsets):
    v = np.einsum("pij,j->pi", g[:, joint], offsets[child])
    return v / np.linalg.norm(offsets[child])


def _elevation(v):
    return np.degrees(np.arcsin(np.clip(v[:, 1], -1, 1)))


def _p(values, qs=(10, 50, 90)):
    return [round(float(v), 1) for v in np.percentile(values, qs)]


def evaluate(recording, model_id, tag, times):
    out_dir = paths.recordings / "action_eval_20261119"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = replay(paths.recordings / recording, model_id=model_id,
                    start_context_frames=60, start_offset_frames=60)
    offsets = LyingCatContact().offsets
    report = {"recording": recording, "model": model_id}
    for stage, local in (("network", result.raw_local), ("final", result.final_local)):
        m = metrics(result, local)
        g = np.asarray([np.stack(_globals(p)) for p in local])
        elbows = [_angle(_bone(g, u, f, offsets), _bone(g, f, w, offsets)) for u, f, w in ((16, 18, 20), (17, 19, 21))]
        report[stage] = {
            "boneErrDeg": m["boneErrDeg"], "jitterStepDeg": m["jitterStepDeg"], "kneeFlexDeg": m["kneeFlexDeg"],
            "thighElevationDeg": {"L": _p(_elevation(_bone(g, 1, 4, offsets))), "R": _p(_elevation(_bone(g, 2, 5, offsets)))},
            "upperArmElevationDeg": {"L": _p(_elevation(_bone(g, 16, 18, offsets))),
                                     "R": _p(_elevation(_bone(g, 17, 19, offsets)))},
            "elbowBendDeg": {"L": _p(elbows[0]), "R": _p(elbows[1])},
            "spineBendDeg": _p(_angle(g[:, 0] @ UP, g[:, 9] @ UP)),
        }
    diags = [d["deadbugLegs"].get("lyingcat") for d in result.diagnostics]
    diags = [d for d in diags if d]
    if diags:
        spine = np.array([d["spineDeg"] for d in diags])
        errors = {}
        for d in diags:
            for key, value in d.get("afterErrorCm", {}).items():
                errors.setdefault(key, []).append(abs(value))
        report["rule"] = {
            "spineDeg": _p(spine), "spineSaturatedFrac": round(float(np.mean(np.abs(spine) > 0.97 * MAX_SPINE_DEG)), 3),
            "supportWeightMean": {k: round(float(np.mean([d["weights"][k] for d in diags])), 3) for k in diags[0]["weights"]},
            "afterErrorAbsCm": {k: _p(v, (50, 90)) for k, v in errors.items()},
        }
    (out_dir / f"lyingcat_rule_{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    solver = LyingCatContact()
    fig, axes = plt.subplots(2, len(times), figsize=(4 * len(times), 7))
    for column, seconds in enumerate(times):
        index = min(int(seconds * 60 / result.infer_every), len(result.final_local) - 1)
        for row, (name, local) in enumerate((("network", result.raw_local[index]), ("final", result.final_local[index]))):
            p = solver._positions(_globals(local.astype(np.float64)))
            head = p[15] - p[0]
            head[1] = 0.0
            head /= max(np.linalg.norm(head), 1e-9)
            x, y = p @ head, p[:, 1]
            ax = axes[row, column]
            for joint in range(1, 22):
                parent = PARENTS[joint]
                color = "r" if joint in (16, 17, 18, 19, 20, 21) else ("b" if joint in (1, 2, 4, 5, 7, 8, 10, 11) else "k")
                ax.plot([x[joint], x[parent]], [y[joint], y[parent]], color + "-", lw=2)
            ax.axhline(float(y[[4, 5]].mean()), color="g", ls=":", lw=1)
            ax.set_aspect("equal")
            ax.set_xlim(-0.8, 0.9)
            ax.set_ylim(-0.9, 0.5)
            ax.set_title(f"{name} t={seconds}s")
    plt.tight_layout()
    image = out_dir / f"lyingcat_rule_{tag}.png"
    plt.savefig(image, dpi=70)
    return report, image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", nargs="?", default="unity_20261116_195301.jsonl")
    parser.add_argument("--model", default=ACTION_RUNTIME_MODELS["lyingcat"])
    parser.add_argument("--tag", default="current")
    parser.add_argument("--times", default="2,5,10,14")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    report, image = evaluate(args.recording, args.model, args.tag, [float(t) for t in args.times.split(",")])
    print(json.dumps({k: report[k] for k in ("final", "rule") if k in report}, ensure_ascii=False))
    print(image)


if __name__ == "__main__":
    main()
