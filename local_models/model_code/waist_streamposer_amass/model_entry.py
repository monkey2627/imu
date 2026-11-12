from pathlib import Path
import sys

MODEL_ID = "waist_streamposer_amass"
ARCHITECTURE = "StreamPoser"
MODEL_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = MODEL_ROOT / "runtime"
CHECKPOINT_ROOT = MODEL_ROOT.parents[1] / "checkpoints"
CHECKPOINT_DIR = CHECKPOINT_ROOT / MODEL_ID

def load():
    runtime_path = str(RUNTIME_ROOT)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    model_path = str(MODEL_ROOT)
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    from model_backend import resolve_waist_model
    return resolve_waist_model(MODEL_ID, CHECKPOINT_ROOT)
