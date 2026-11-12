# Repository Guidelines

## Project Structure

This is a Unity 2022.3.62f3 project. Runtime gameplay and motion code lives mainly in `Assets/Scripts/`, with rhythm-game logic in `Assets/Scripts/RhythmMode/`. Feature-specific scripts, builders, and assets are grouped under `Assets/RingBridge/`, `Assets/RingSkinC/`, `Assets/TaikoSelf/`, and related feature folders. Editor tools and verification are in `Assets/Scripts/Editor/` and feature `Editor/` folders. Production scenes are in `Assets/Scenes/`; recorded motion, audio, models, and UI assets are under `Assets/`. Unity settings and package declarations are in `ProjectSettings/` and `Packages/`. `output/` contains verification artifacts; do not treat it as source.

## Build and Development

Open the repository root in Unity Hub with the pinned Unity version from `ProjectSettings/ProjectVersion.txt`. Use **File > Build Settings** to select a registered scene and build target. There is no repository-level CLI build or test script. Avoid committing generated `Library/`, `Temp/`, `Build/`, or IDE files; they are ignored.

## Coding Style and Assets

Follow surrounding C# style: four-space indentation, PascalCase for types and public members, and descriptive camelCase for local variables and private fields. Keep runtime code separate from Unity Editor-only tooling under `Editor/`. Preserve Unity `.meta` files and GUIDs when adding, moving, or renaming assets; make asset changes through Unity when practical. Keep scene and asset names descriptive and consistent with existing exercise/mode names (for example, `SideBridge_RingBridge_Dynamic.unity`).

## Testing and Verification

Run the relevant Unity Editor checks from the `Tools` menu after changing gameplay, scene setup, motion, or device feedback. Useful checks documented in `README.md` include `Tools > Ring Bridge > Verify New Scoring and 30 Second Scenes`, `Tools > Ring Skin C > Verify HIT-only Tracker Requests`, and `Tools > Recorded Coach > Verify Recorded Bone Orientation`. Some checks write reports under `output/`; inspect results and avoid committing incidental outputs unless they are intentional. Hardware behavior still requires device testing.

## Commits and Pull Requests

Recent commits use short, imperative subject lines (for example, `Show gold progress lightbar during bridge holds`). Keep each commit focused and describe the behavior changed. Pull requests should summarize affected scenes/features, list verification performed and results, link related issues when available, and include screenshots or a short recording for visible scene/UI changes. Call out any hardware-only validation that remains.

## Configuration and Safety

Do not commit credentials, local device configuration, or generated Unity/IDE state. Preserve scene references and asset metadata when editing serialized Unity content.
