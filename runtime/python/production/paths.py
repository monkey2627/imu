"""Filesystem locations owned by the production bridge."""

import os
from pathlib import Path


class paths:
    runtime_root = Path(__file__).resolve().parents[1]
    workspace_root = runtime_root.parents[1]
    checkpoint = Path(os.environ.get("MOTION_RECOVER_CHECKPOINT_DIR", runtime_root / "checkpoints"))
    waist_checkpoint = Path(
        os.environ.get(
            "MOTION_RECOVER_WAIST_CHECKPOINT_DIR",
            workspace_root / "local_models" / "checkpoints",
        )
    )
    recordings = Path(os.environ.get("MOTION_RECOVER_RECORDINGS_DIR", runtime_root / "recordings"))
    smpl_file = runtime_root / "mobileposer" / "smpl" / "basicmodel_m.pkl"

    # Kept as a neutral alias for tools that need the Python runtime root.
    root_dir = runtime_root
