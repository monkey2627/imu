#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unity", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    import mobileposer.test_ours_surface as adapter
    adapter.FPS = 60
    report = adapter.prepare(args.capture, args.output)
    desired = report["layouts"]["wrists_shanks_waist"]
    if desired["status"] != "prepared":
        raise RuntimeError(f"five physical sensors unavailable: {desired}")
    ref, ids, meta = adapter.load_reference(args.capture, args.output, args.unity)
    report["reference"] = meta
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    from mobileposer.smoke_finetune_ours import make_targets
    targets, _ = make_targets(args.output, args.output / "supervision")
    input_data = torch.load(desired["input"], map_location="cpu", weights_only=False)
    assert input_data["fps"] == 60
    assert input_data["layout_labels"] == ["lw", "rw", "ls", "rs", "waist"]
    assert torch.equal(input_data["timestamps"], targets["timestamps"])
    assert len(input_data["acc"][0]) == len(targets["pose"])
    assert np.isfinite(ref).all()

    summary = {
        "status": "ok",
        "capture": str(args.capture.resolve()),
        "fps": 60,
        "frames": len(targets["pose"]),
        "duration_s": float(input_data["timestamps"][-1] - input_data["timestamps"][0]),
        "layout": "wrists_shanks_waist",
        "labels": input_data["layout_labels"],
        "sensor_diagnostics": {k: report["sensors"][k] for k in input_data["layout_labels"]},
        "input": desired["input"],
        "targets": str((args.output / "supervision" / "targets.pt").resolve()),
        "supervision_warning": "motion.json/SlimeVR avatar pseudo-labels; same-sequence personalization, not independent mocap ground truth",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
