"""Train ONLY the Poser module with the kinematic-tree head + local-rotation
loss (models/poser_hier.py), on the wrists_shanks_waist layout data, with the
same hyperparameters as the baseline Poser (train_hypers, 60 epochs).

Run from SparseIMU/ (env: mobileposer):
    MOBILEPOSER_PROCESSED_DATASETS=data/no_head_5imu_surface_processed/wrists_shanks_waist \
    MOBILEPOSER_CHECKPOINT_DIR=checkpoints/poser_hier_wsw \
    MOBILEPOSER_TRAIN_COMBOS=all_5imu MOBILEPOSER_DISABLE_WANDB=1 \
    python -m mobileposer.train_poser_hier [--fast-dev-run] [--local-weight 1.0]
"""
from argparse import ArgumentParser

from lightning.pytorch import seed_everything

from mobileposer.config import paths
from mobileposer.models.poser_hier import PoserHier
from mobileposer.train import TrainingManager, get_checkpoint_path

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--fast-dev-run', action='store_true')
    ap.add_argument('--local-weight', type=float, default=1.0)
    ap.add_argument('--flat', action='store_true', help='keep flat head, only add the local loss')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed, workers=True)
    paths.checkpoint.mkdir(parents=True, exist_ok=True)
    ckpt_path = get_checkpoint_path(None, 'scratch')
    TrainingManager(finetune=None, fast_dev_run=args.fast_dev_run).train_module(
        PoserHier(local_weight=args.local_weight, flat=args.flat), 'poser', ckpt_path)
