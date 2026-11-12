"""Offline pose-level bridge from a SLAM CSV to CameraFusionGRU.

This module deliberately keeps the hardware boundary small: a device-specific
collector only needs to produce the documented SLAM CSV. The body-IMU input is
still supplied in the project's existing processed ``.pt`` format for now.
"""

import argparse
import csv
from pathlib import Path

import torch

from mobileposer.camera_fusion import CameraFusionGRU
from mobileposer.evaluate_no_head_5imu import prepare_imu
from mobileposer.utils.model_utils import load_model


def read_slam_csv(path: Path) -> torch.Tensor:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "x", "y", "z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing SLAM CSV columns: {sorted(missing)}")
        for row in reader:
            if int(row.get("tracking_state", 1)) == 0:
                raise ValueError("SLAM CSV contains lost tracking; split or repair the segment first")
            rows.append([float(row[axis]) for axis in ("x", "y", "z")])
    if not rows:
        raise ValueError("SLAM CSV contains no samples")
    pose = torch.tensor(rows, dtype=torch.float32)
    return pose - pose[:1]


def run(base_checkpoint: Path, fusion_checkpoint: Path, sequence_file: Path,
        slam_csv: Path, output: Path, sequence_index: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(sequence_file, map_location="cpu", weights_only=False)
    acc = data["acc"][sequence_index]
    ori = data["ori"][sequence_index]
    imu = prepare_imu(acc, ori).to(device)

    base = load_model(str(base_checkpoint)).to(device).eval()
    with torch.no_grad():
        base.reset()
        _, _, imu_pos, _ = base.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
        imu_pos = imu_pos.squeeze(0)

    slam_pos = read_slam_csv(slam_csv)
    count = min(len(imu_pos), len(slam_pos))
    imu_pos = imu_pos[:count]
    slam_pos = slam_pos[:count].to(device)

    blob = torch.load(fusion_checkpoint, map_location=device, weights_only=False)
    manifest = blob.get("manifest", {})
    fusion = CameraFusionGRU(hidden=int(manifest.get("hidden", 32))).to(device)
    fusion.load_state_dict(blob.get("state_dict", blob))
    fusion.eval()
    with torch.no_grad():
        fused, weight, _ = fusion(imu_pos.unsqueeze(0), slam_pos.unsqueeze(0))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fused_position": fused.squeeze(0).cpu(),
                "imu_position": imu_pos.cpu(),
                "slam_position": slam_pos.cpu(),
                "imu_weight": weight.squeeze(0).cpu()}, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--slam-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-index", type=int, default=0)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
