# MotionRecover Workspace

This workspace combines the locally maintained runtime projects with independently packaged model code. It does not depend on the Git history of the older working directories.

## Layout

- `runtime/python/`: Python production runtime, analysis tools, configuration, recordings, Android APK, and local runtime checkpoints copied from `realUse/`.
- `runtime/slimevr/SlimeVR-Server/`: source copied from `C:\Users\admin\Desktop\SlimeVR-Server`; generated `build/`, `.gradle/`, `.kotlin/`, and `node_modules/` are omitted.
- `runtime/unity/SpineRecovery_V2/`: source copied from the local Unity project; generated `Library/`, `Temp/`, `Logs/`, `obj/`, and user-local settings are omitted.
- `runtime/exports/`: destination for generated Unity/runtime exports before any intentional server upload.
- `local_models/checkpoints/`: the 20 runtime model packages, arranged by model ID, including each IMUDINO head and its required backbone.
- `local_models/model_code/`: 17 side-by-side model directories. Every directory contains its own loader, runtime/network implementation, training package and entrypoints, data-processing scripts, and README; model directories do not import network or training code from one another.
- `sync/`: source-of-truth map, manifests, and synchronization procedures.

PNP remains an optional comparison baseline and is not part of the 20-model runtime catalog.

The five-IMU IMUDINO deployment, launch command, input/root contract and verification are documented in [`docs/IMUDINO_REALUSE.md`](docs/IMUDINO_REALUSE.md). Its training source is the separate server repository `/home/duanyuhan/dyh/motion/IMUDINO-Release`.

For the training data-generation and dataset/window-processing code path, see [`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md).

## Server training workspace

The training data and generated training artifacts remain under `/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030/` on `10.132.25.165`. The code needed for each retained checkpoint is stored inside its corresponding `local_models/model_code/<model_id>/` directory. Those model packages retain server paths where required but do not depend on a workspace-level shared training tree. Synchronization is explicit; no automatic sync is configured.

This workspace is the authoritative local source. See `sync/source-of-truth.md` before synchronizing server data or exported models.

The 20 realtime model IDs, their checkpoints, model implementations, data preparation, and training methods are indexed in `local_models/README.md` and `local_models/model_code/README.md`.

## Local validation

Test-only source and temporary artifacts were removed on 2026-11-11 in favor of the actual production and training paths. Validation should now use the real SlimeVR → Python → Unity workflow and the documented training/evaluation entrypoints.
