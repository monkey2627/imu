"""Canonical five-IMU placement for line14.

The order is fixed to the PNP six-sensor setup with the head sensor omitted:
left/right lower arm, left/right lower leg, pelvis.  Do not use the historical
waist-at-spine1 or alternative five-sensor layouts for line14 training.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


PNP5_DATA_ROOT = Path("data/pnp5_60a/pnp5")
PNP5_LABELS = ("lw", "rw", "ls", "rs", "pelvis")
PNP5_SMPL_JOINTS = (18, 19, 4, 5, 0)
PNP5_SURFACE_VERTICES = (1961, 5424, 1176, 4662, 3021)
# PNP's test-pack sensor order is lower arms, lower legs, head, pelvis.
PNP6_KEEP_INDICES = (0, 1, 2, 3, 5)
PNP5_FPS = 60


def pnp5_layout_metadata() -> dict[str, object]:
    return {
        "name": "PNP5_without_head",
        "sensor_count": 5,
        "labels": list(PNP5_LABELS),
        "smpl_joints": list(PNP5_SMPL_JOINTS),
        "surface_vertices": list(PNP5_SURFACE_VERTICES),
        "pnp6_test_indices": list(PNP6_KEEP_INDICES),
        "fps": PNP5_FPS,
    }


def assert_pnp5_data_root(root: str | Path) -> Path:
    """Reject old-layout folders at line14 training entry points."""
    path = Path(root).resolve()
    expected = (Path(__file__).resolve().parents[2] / PNP5_DATA_ROOT).resolve()
    if path != expected:
        raise ValueError(
            f"line14 sensor placement is fixed to {expected}: "
            "[lw, rw, ls, rs, pelvis] at SMPL joints [18, 19, 4, 5, 0]. "
            f"Got {path}."
        )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def validate_pnp5_dataset_metadata(root: str | Path) -> None:
    """Check one small cached subset so a renamed/replaced old layout cannot pass."""
    root = assert_pnp5_data_root(root)
    candidates = [root / "SSM_synced.pt", *sorted(root.glob("*.pt"))]
    sample = next((path for path in candidates if path.is_file()), None)
    if sample is None:
        raise FileNotFoundError(f"no processed .pt datasets in {root}")
    data = torch.load(sample, map_location="cpu", weights_only=False)
    labels = tuple(data.get("layout_labels", ()))
    joints = tuple(int(j) for j in data.get("layout_joints", ()))
    synthesis = data.get("synthesis") or {}
    if data.get("layout") != "pnp5" or labels != PNP5_LABELS or joints != PNP5_SMPL_JOINTS:
        raise ValueError(
            f"{sample} is not the canonical PNP5 layout: "
            f"layout={data.get('layout')!r}, labels={labels}, joints={joints}"
        )
    if synthesis.get("fps") != PNP5_FPS or synthesis.get("mode") != "surface":
        raise ValueError(f"{sample} must contain 60-Hz PNP surface synthesis metadata")
    attachments = (synthesis.get("attachments") or {}).get("attachments", {})
    vertices = tuple(
        int(attachments[label]["vertices"][attachments[label]["weights"].index(1.0)])
        for label in PNP5_LABELS
    )
    if vertices != PNP5_SURFACE_VERTICES:
        raise ValueError(f"{sample} has non-PNP surface positions: {vertices}")


def validate_pnp5_surface_config(config_path: str | Path | None = None) -> None:
    """Fail if the configured synthesis attachments drift from the fixed layout."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "pnp_surface_imu.json"
    config = json.loads(Path(config_path).read_text())
    attachments = config["attachments"]
    joints = tuple(int(attachments[label]["joint"]) for label in PNP5_LABELS)
    vertices = tuple(int(attachments[label]["vertices"][attachments[label]["weights"].index(1.0)])
                     for label in PNP5_LABELS)
    if joints != PNP5_SMPL_JOINTS or vertices != PNP5_SURFACE_VERTICES:
        raise ValueError(
            "pnp_surface_imu.json no longer matches the canonical line14 layout: "
            f"joints={joints}, vertices={vertices}"
        )
