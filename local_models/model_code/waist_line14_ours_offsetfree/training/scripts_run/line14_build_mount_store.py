"""Build an IMU store with per-identity position-only mount (lever-arm) offsets (rot_deg=0) for line 14 / E5.
usage: python scripts_run/line14_build_mount_store.py LEVER_CM OUT_DIR [WORKERS]
Labels/joints are the shape-aware FK joints of data/line9_store_shape (symlinked); only IMU acceleration changes."""
import os, sys
os.environ["MOBILEPOSER_FPS"] = "60"; os.environ.setdefault("OMP_NUM_THREADS", "2")
from mobileposer.line9.data import load_identity_records
from mobileposer.line9.calib import build_calib_store
lever, out = float(sys.argv[1]), sys.argv[2]
rec = load_identity_records("data/pnp5_60a/pnp5", "results/line9/data_audit.json", ("train", "validation"))
build_calib_store(rec["train"] + rec["validation"], "data/line9_store_shape", out, 0.0, lever, workers=int(sys.argv[3]) if len(sys.argv) > 3 else 16)
