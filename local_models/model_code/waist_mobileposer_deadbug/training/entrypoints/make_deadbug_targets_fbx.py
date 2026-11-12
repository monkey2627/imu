#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    physical = torch.load(args.input, map_location="cpu", weights_only=False)
    fitted = torch.load(args.fit, map_location="cpu", weights_only=False)
    assert physical["layout"] == "wrists_shanks_waist"
    assert physical["layout_labels"] == ["lw", "rw", "ls", "rs", "waist"]
    assert physical["fps"] == 60
    pose30 = fitted["pose"][0]
    tran30 = fitted["tran"][0]
    shape = fitted["shape"][0]
    meta = fitted["fit_metadata"][0]
    fps30 = float(meta["source_meta"]["fps"])
    assert abs(fps30 - 30.0) < 1e-6

    target_time = physical["timestamps"].double().numpy()
    target_time -= target_time[0]
    source_time = np.arange(len(pose30), dtype=np.float64) / fps30
    if target_time[-1] > source_time[-1] + 1.0 / fps30:
        raise ValueError((target_time[-1], source_time[-1]))
    target_time = np.minimum(target_time, source_time[-1])
    pose60 = np.stack(
        [Slerp(source_time, Rotation.from_matrix(pose30[:, j].numpy()))(target_time).as_matrix()
         for j in range(24)],
        axis=1,
    )
    tran60 = np.stack(
        [np.interp(target_time, source_time, tran30[:, k].numpy()) for k in range(3)],
        axis=1,
    )
    pose60 = torch.tensor(pose60, dtype=torch.float32)
    tran60 = torch.tensor(tran60, dtype=torch.float32)

    from mobileposer.articulate.model import ParametricModel
    body = ParametricModel(Path(__file__).resolve().parent / "pnp_workspace/mobileposer/smpl/basicmodel_m.pkl")
    global_pose, joints = body.forward_kinematics(pose60, shape, tran60)
    target = {
        "pose": pose60,
        "global_pose": global_pose,
        "joint": joints,
        "tran": tran60,
        "shape": shape,
        "timestamps": physical["timestamps"],
        "fps": 60,
        "metadata": {
            "kind": "same-capture FBX-to-SMPL pseudo-labels",
            "independent_ground_truth": False,
            "input_sha256": sha256(args.input),
            "fit_sha256": sha256(args.fit),
            "source_fps": fps30,
            "target_fps": 60,
            "fit_metrics": meta["fit"],
            "resampling": "per-joint rotation SLERP; translation linear interpolation",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "frames": len(pose60),
        "duration_s": float(target_time[-1]),
        "fit_metrics": meta["fit"],
        "sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
