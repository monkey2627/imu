# Training source and data flow

The 16 retained waist-surface models are packaged independently under
`local_models/model_code/<model_id>/`. Each model directory contains:

- `training/mobileposer/`: that model's private network and trainer source;
- `training/entrypoints/`: preparation, training, fine-tuning, export, or evaluation entrypoints used for that checkpoint;
- `data_processing/`: preprocessing and dataset-construction scripts;
- `README.md` and `training/SERVER_TRAINING_NOTES.md`: model-specific method, checkpoint, server path, and data notes.

Model directories do not import training or network code from another model directory. Repeated source is intentional so that each checkpoint can be handed off and modified independently.

## Server boundary

Large training inputs and generated artifacts remain on `10.132.25.165`, primarily under:

`/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030/`

AMASS files, real-use recordings, processed `.pt` datasets, caches, logs, and unselected checkpoints are not mirrored locally. The selected 16 runtime checkpoints are under `local_models/checkpoints/<model_id>/`. No automatic synchronization is configured; see [`../sync/source-of-truth.md`](../sync/source-of-truth.md).

## Removed aggregate workspace

The former workspace-level `training/` mirror was removed on 2026-11-11. It duplicated the model packages and also mixed in obsolete realtime copies, historical renderers, PNP comparisons, and experiments that do not correspond to the retained 16 checkpoints. Training work should start from the relevant model directory, not from a shared aggregate package.
