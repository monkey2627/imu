"""Paths shared by the realtime MobilePose bridge and Web SMPL viewer."""

import os
from pathlib import Path


class paths:
    root_dir = Path(__file__).resolve().parents[1]
    checkpoint = Path(os.environ.get("MOBILEPOSER_CHECKPOINT_DIR", root_dir / "checkpoints"))
    smpl_file = root_dir / "mobileposer/smpl/basicmodel_m.pkl"
