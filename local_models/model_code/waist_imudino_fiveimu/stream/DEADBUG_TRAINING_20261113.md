# AMASS + DeadBug streaming model

The server original launches with `bash run.sh` (GPU 0 by default; override with `DEADBUG_GPU`).
This local audit mirror keeps that script as `run_deadbug_20261113.sh`.
This experiment follows `imudino_wristankle_20261112/run_limbmix.sh`:
18 AMASS datasets plus DeadBug and DeadBugMix, frozen IMUDINO backbone,
the same stateful StreamHead, 60,000 steps, batch 64, crop 360, warmup 60,
AdamW lr 0.001, velocity loss weight 1, seed 42, validation every 2,000 steps,
exercise sampling fraction 0.10. Training starts from a fresh head.

Source: `data_system/runtime/output/training_data/DeadBug` (DuoMo pseudo-labels,
224 clips from 36 source videos). The alternative DeadBugFlowHMR export is
not included. AMASS streaming caches are reused through symlinks; BirdDog
and BirdDogMix are excluded from this run.

Data augmentation preserves the prior upper/lower-body decoupling:

- `cross`: trunk/arms/root from training clip A, leg joints from training clip B
  belonging to another source video.
- `armonly`: A's arm motion with leg joints fixed at A's tabletop rest frame.
- `legonly`: A's leg motion with arm joints fixed at A's arms-up rest frame.

This is data recombination, using the same single full-pose streaming head as
the BirdDog run. After changing poses, all IMUs and joints are re-synthesized
with the identical 60 Hz wrist/ankle/waist surface attachment configuration
(`0e9da789a72715b2cc165bb65fc0ac285b5a8334ff1b8f6f3e25fc9adce1c338`).
The BirdDog kneeling rest rule is replaced with supine, world-up limb scoring;
clips without a suitable rest frame (mean direction error >35 degrees) skip
that augmentation. Knees near 90 degrees are preferred for tabletop legs.

Original DeadBug clips are grouped by `DeadBug:<source_video_sha256>` and
split by the existing deterministic 10% hash rule (seed 42). Recombination
uses only training videos, and every DeadBugMix sequence is train-only.
The held-out DeadBug videos remain unmodified. See
`data/processed/wrists_shanks_waist/split_report.json` for actual counts.

Outputs:

- `logs/synth.log`, `logs/prepare.log`, `logs/train.log`
- `logs/PREPARED`, `logs/COMPLETE` (or `logs/FAILED`)
- `runs/stream_deadbug_limbmix/{best.pt,latest.pt,metrics.jsonl,COMPLETE}`

The existing trainer now accepts optional `--special-prefix` and
`--special-name` arguments, allowing DeadBug oversampling and a `deadbug`
validation metric. Its BirdDog defaults remain the same. Checkpoint selection
keeps the prior mean of AMASS and exercise validation angle errors.
