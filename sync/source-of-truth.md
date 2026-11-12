# Source of Truth and Sync Boundaries

Git history is intentionally not used to decide which copy is authoritative.

| Component | Authoritative source | Workspace copy | Rule |
|---|---|---|---|
| Python runtime and analysis | `runtime/python/` | Same | Workspace source is authoritative. |
| SlimeVR | `runtime/slimevr/SlimeVR-Server/` | Same | Workspace source is authoritative. It was verified byte-for-byte against the former desktop source on 2026-11-11; rebuild generated dependencies locally. |
| Unity | `runtime/unity/SpineRecovery_V2/` | Same | Workspace source is authoritative. Open it with Unity 2022.3.62f3c1 to regenerate `Library/`, IDE projects, and user-local settings. |
| Retained model code | The code packaged with each selected checkpoint | `local_models/model_code/<model_id>/` | Each of the 17 models owns a private runtime/network, training, and data-processing copy. Do not reintroduce a shared workspace-level training package. |
| Training data | Server `data/`, `pnp_workspace/data/`, `line14_workspace/`, `deadbug60/`, and related training inputs | No full local copy | Server-only. Do not mirror datasets or generated processed `.pt` files. |
| Training outputs and caches | Server `checkpoints/`, `logs/`, `artifacts/`, `cache/`, `smoke/` | No training-output copy | Server-only by default. Export only selected validated checkpoints and summaries into `local_models/` or `sync/`. |

The runtime uses two checkpoint groups: 17 model IDs (the original 16 waist-surface models plus IMUDINO) are centralized in `local_models/checkpoints/<model_id>/`, and the two legacy PNP models remain under `runtime/python/checkpoints/<model_id>/`. Each model ID has an independent directory in `local_models/model_code/<model_id>/` containing private runtime/network, training, and data-processing code. The production application runs from the architecture-neutral `runtime/python/production/` package; model algorithms connect only through `production/model_backends.py`. Model-package copies are independently inspectable handoff snapshots.

## Server paths

- IMUDINO training authority: `/home/duanyuhan/dyh/motion/IMUDINO-Release/`. The selected five-IMU `best.pt`, required `imudino.pt`, and private source snapshot were exported read-only on 2026-11-11 into model ID `waist_imudino_fiveimu`; training data, caches and renderings remain on the server. See `../docs/IMUDINO_REALUSE.md` and its deployment manifest. This adds a 17th model package to the original 16-model catalog.

- Training workspace: `/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030/`
- Python Git checkout (not the training workspace): `/home/duanyuhan/dyh/motion/realUse/`

The server workspace also contains large `data/`, `cache/`, `pnp_workspace/`, and `line14_workspace/` directories. Required source/config/model assets are preserved inside the relevant model packages; data subdirectories and other large training inputs remain on the server. The separately named `data/` at the training root is not the only location containing training inputs.

These are different server directories. The Python checkout is not the destination for training code synchronization.

## Copy rules

- No automatic two-way synchronization is configured.
- Never run mirror/delete synchronization against the server `data/` directory.
- For code refreshes, copy into a staging directory, compare file counts and SHA-256 manifests, then promote after review.
- For model export, record source checkpoint path, training configuration, dataset/version, timestamp, and SHA-256 alongside the exported model.
- Do not include Unity/SlimeVR build caches in source manifests; regenerate them with the project's documented toolchain.
- Local runtime and Unity exports go under `runtime/exports/`; upload them to the server only when they are selected as training inputs.

## Initial assembly record (2026-11-10)

- Runtime Python, runtime checkpoints, SlimeVR source, and Unity source were copied from the local authoritative paths listed above.
- Unity generated/cache directories and SlimeVR generated build/dependency directories were excluded from the new workspace.
- The server training source was initially mirrored and verified, then split into the 16 independent model packages. The redundant aggregate mirror, obsolete realtime copies, comparison experiments, and historical renderers were removed on 2026-11-11.
- Server data, caches, unselected training checkpoints, logs, and generated artifacts were not copied.
- Python syntax validation was completed during assembly. Test-only source was later removed in favor of actual production and training workflows.
- The old sibling directories `realUse/`, `SlimeVR-Server/`, `SpineRecovery_V2/`, and `server_training_20261030/` were retired after source comparison and consolidation on 2026-11-11. `MotionRecoverWorkspace/` is now the sole local working root.

## MobilePoser source completion (2026-11-11)

- SSH access to `10.132.25.165` was re-established and the active training package was identified at `realuse_waist_surface_20261030/pnp_workspace/mobileposer/` (164 Python source files).
- The required source package, configs, schema sources, and `smpl/basicmodel_m.pkl` were incorporated into the applicable independent model packages; Python bytecode caches are excluded.
- This fills the implementation gap: `run_preprocess.sh` invokes `mobileposer.no_head_layouts` (synthesis in `_synthesize`, with surface kinematics in `surface_imu.py`), while `train_mobileposer_module.py` calls `mobileposer.train.TrainingManager`, which obtains `PoseDataModule`/`PoseDataset` from `mobileposer/data.py`.
- Training data, processed `.pt` files, checkpoints, caches, logs, and output artifacts were not copied. Existing server paths/environment assumptions remain unchanged.
