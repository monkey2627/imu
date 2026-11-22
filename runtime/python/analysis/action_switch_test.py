"""Action-switch test of the realtime bridge (production/run.py), without Unity or SlimeVR.

run.main_async runs unchanged; only the SolarXR feed is replaced by a 60 Hz feeder that plays the
sensor ticks of real recordings whose start-pose calibration passed.  The test talks to the bridge
over its WebSocket exactly like Unity (actionSelect, exerciseCalibrationStart, startRecording,
stopRecording) and never restarts Python between actions.

Checks, per action switch:
  * the first output after actionSelect already belongs to the new action (actionId, model, rules),
    with the previous start pose / constraint / contact state gone and a fresh model session;
  * after the calibration the new action's start pose is active (actionState.startPose, constraint);
  * a second set (startRecording after stopRecording) re-arms the start pose;
  * every recording written by the bridge is reproduced by production.live_replay from a FRESH
    state (modelSessionFresh): nothing from the previous action influences the output.

    python -m analysis.action_switch_test [--sequence lyingcat,hipthrust,lyingcat,deadbug,lyingcat]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import websockets

from production import input_recording, run
from production.input_recording import _sample_for
from production.model_backends import ACTION_RUNTIME_MODELS

RECORDINGS = Path(__file__).resolve().parents[1] / "recordings"
SOURCES = {     # real recordings whose live start-pose calibration completed (calibration starts at t=0)
    "lyingcat": ("lyingcat_20261120_230158.jsonl", "quadruped"),
    "hipthrust": ("hipthrust_20261119_234600.jsonl", "supine"),
    "deadbug": ("deadbug_20261119_234510.jsonl", "tabletop"),
    "sidebridge_left": ("sidebridge_left_20261122_033550.jsonl", "side_rest"),
    # Unity's bird-dog flow (not ExerciseCalibration): four-point hold 0-7.3 s, then birdDogBaseCalibrate.
    "birddog": ("birddog_20261123_005257.jsonl", "quadruped"),
}
BIRDDOG_HOLD_SECONDS = 7.3
RULE_KEYS = {"lyingcat": "lyingcat", "hipthrust": "bridge", "deadbug": "deadbugFloor", "sidebridge_left": "sidebridge",
             "birddog": "birddog"}
PORT = 21291


def load_segment(name, label_to_key):
    rows = [json.loads(line) for line in (RECORDINGS / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    frames = []
    for row in rows:
        if row.get("type") != "tick":
            continue
        update = {label_to_key[label]: _sample_for(raw, label_to_key[label]) for label, raw in row["samples"].items()}
        frames.extend([update] * max(1, int(row.get("ticks", 1))))
    return frames


class Feeder:
    """Fake SolarXR: one update per 60 Hz tick from the current segment (holds the last one at the end)."""

    def __init__(self, frames):
        self.frames, self.index = frames, 0

    def play(self, frames):
        self.frames, self.index = frames, 0

    async def stream(self, minimum_time_since_last_ms=16):
        start = time.perf_counter()
        tick = 0
        while True:
            delay = start + tick / 60.0 - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            frame = self.frames[min(self.index, len(self.frames) - 1)]
            self.index += 1
            tick += 1
            yield dict(frame)


def bridge_args(port):
    return argparse.Namespace(
        host="127.0.0.1", port=port, web=False, web_host="127.0.0.1", web_port=8765, calibration_frames=0,
        acceleration_source="server", checkpoint_root=run.DEFAULT_CHECKPOINT_ROOT, model=run.DEFAULT_RUNTIME_MODEL,
        waist_checkpoint_root=run.WAIST_CHECKPOINT_ROOT, checkpoint=None, anchor="auto", infer_every=2,
        output_fps=None, include_contact=False, contact_interval=30, sync_calibration=False,
        upright_calibration_frames=30, upright_settle_seconds=3.0, upright_calibration=False,
        skip_upright_calibration=False, record=None, no_auto_compare=True, no_pose_calibration=False,
        anchor_sensed=False, no_arm_prior=False, deadbug_legs="auto", model_accel_fix=False, torch_threads=8,
        feed_interval_ms=16, replay=None, config=None)


class Client:
    def __init__(self, socket):
        self.socket, self.frames = socket, []

    async def receive(self):
        async for message in self.socket:
            frame = json.loads(message)
            if "diagnostics" in frame or "exerciseCalibration" in frame:
                self.frames.append(frame)

    async def send(self, **control):
        await self.socket.send(json.dumps(control))

    def diag(self, frame):
        return frame.get("diagnostics") or frame

    async def wait(self, predicate, timeout, after=0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            for frame in self.frames[after:]:
                if predicate(self.diag(frame)):
                    return self.diag(frame)
            await asyncio.sleep(0.05)
        raise TimeoutError("condition not reached")


def check(results, ok, message):
    results.append((bool(ok), message))
    print(("PASS " if ok else "FAIL ") + message, flush=True)


async def scenario(sequence, results):
    first = json.loads((RECORDINGS / SOURCES[sequence[0]][0]).read_text(encoding="utf-8").splitlines()[0])
    label_to_key = dict(first["labelToTrackerKey"])
    segments = {action: load_segment(SOURCES[action][0], label_to_key) for action in set(sequence)}
    feeder = Feeder(segments[sequence[0]])
    run.stream_trackers = feeder.stream
    run._code_stamp = lambda: 0.0
    bridge = asyncio.create_task(run.main_async(bridge_args(PORT)))
    socket = None
    for _ in range(600):
        try:
            socket = await websockets.connect(f"ws://127.0.0.1:{PORT}", max_size=None)
            break
        except OSError:
            if bridge.done():
                bridge.result()
            await asyncio.sleep(0.2)
    client = Client(socket)
    receiver = asyncio.create_task(client.receive())
    await client.wait(lambda d: "actionState" in d, 120)
    previous = None
    for step, action in enumerate(sequence):
        start_pose = SOURCES[action][1]
        print(f"\n=== step {step + 1}: {previous} -> {action} (no Python restart) ===", flush=True)
        mark = len(client.frames)
        await client.send(command="actionSelect", action=action)          # Unity: action chosen
        d = await client.wait(lambda d: (d.get("actionState") or {}).get("action") == action
                              and d.get("modelInputFrame", -1) >= 0, 60, mark)
        state, legs = d["actionState"], d.get("deadbugLegs") or {}
        check(results, d["actionId"] == action and d["model"] == ACTION_RUNTIME_MODELS[action],
              f"{action}: first output after switch has actionId={d['actionId']} model={d['model']}")
        if previous is not None:
            reset = state.get("lastReset") or {}
            check(results, reset.get("previousAction") == previous and reset.get("modelSessionReset")
                  and reset.get("historyBufferCleared"),
                  f"{action}: lastReset previousAction={reset.get('previousAction')} "
                  f"previousStartPose={reset.get('previousStartPose')} sessionReset={reset.get('modelSessionReset')}")
        check(results, state["startPose"] is None and not state["startPoseConstraint"]["enabled"]
              and state["startPoseConstraint"]["action"] is None,
              f"{action}: before calibration no start pose / constraint inherited ({state['startPoseConstraint']})")
        stale = [key for other, key in RULE_KEYS.items() if other != action and key in legs]
        check(results, not stale, f"{action}: no other action's rule in the output (found {stale})")

        mark = len(client.frames)
        feeder.play(segments[action])                                       # user holds the start pose
        if action == "birddog":
            # Unity's bird-dog calibration: session start, four-point hold, base command, session end.
            await client.send(command="calibrationSessionStart", name=action)
            await client.send(command="birdDogStartCalibrate")
            await asyncio.sleep(BIRDDOG_HOLD_SECONDS)
            await client.send(command="birdDogBaseCalibrate")
            await client.send(command="calibrationSessionEnd")
            d = await client.wait(lambda d: "start pose fit (birddog/quadruped)" in (d.get("controlMessage") or ""),
                                  20, mark)
            check(results, "模型已从 quadruped" in d["controlMessage"],
                  f"{action}: calibration ({d['controlMessage'][:120]})")
        else:
            await client.send(command="exerciseCalibrationStart", action=action, startPose=start_pose)
            d = await client.wait(lambda d: (d.get("exerciseCalibration") or {}).get("action") == action
                                  and (d.get("exerciseCalibration") or {}).get("outcome") in ("complete", "failed"),
                                  60, mark)
            outcome = d["exerciseCalibration"]["outcome"]
            check(results, outcome == "complete", f"{action}: calibration {outcome} "
                  f"({(d['exerciseCalibration'].get('results') or [''])[-1][:120]})")
        d = await client.wait(lambda d: (d.get("actionState") or {}).get("startPose") == start_pose, 10, mark)
        state, legs = d["actionState"], d.get("deadbugLegs") or {}
        constraint = legs.get("startPoseConstraint") or {}
        check(results, state["action"] == action and constraint.get("action") == action
              and constraint.get("startPose") == start_pose,
              f"{action}: first output after calibration: session {state['calibrationSessionId']}, "
              f"startPose={state['startPose']}, constraint enabled={constraint.get('enabled')} "
              f"weights={constraint.get('weights')} armWeights={constraint.get('armWeights')}")
        if action == "deadbug":
            check(results, constraint.get("enabled") and constraint.get("weights") == [1.0, 1.0],
                  "deadbug: tabletop constraint starts at full weight")
        stale = [key for other, key in RULE_KEYS.items() if other != action and key in legs]
        check(results, not stale and RULE_KEYS[action] in legs,
              f"{action}: only this action's rule runs (has {RULE_KEYS[action]}: {RULE_KEYS[action] in legs}, "
              f"stale {stale})")
        await asyncio.sleep(4.0)
        await client.send(command="stopRecording")                          # Unity: set ended
        await asyncio.sleep(0.5)
        # A second set of the same action: the start pose is re-armed for the new recording.
        mark = len(client.frames)
        feeder.play(segments[action])
        await client.send(command="startRecording", name=f"{action}_set2")
        d = await client.wait(lambda d: (d.get("recordingFile") or "").startswith(f"{action}_set2"), 10, mark)
        constraint = (d.get("deadbugLegs") or {}).get("startPoseConstraint") or {}
        check(results, d["actionState"]["startPose"] == start_pose and constraint.get("startPose") == start_pose
              and (action != "deadbug" or constraint.get("enabled")),
              f"{action}: second set re-arms start pose {constraint.get('startPose')} "
              f"(enabled={constraint.get('enabled')})")
        await asyncio.sleep(2.0)
        await client.send(command="stopRecording")
        await asyncio.sleep(0.5)
        previous = action
    receiver.cancel()
    await socket.close()
    bridge.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", default="lyingcat,hipthrust,lyingcat,deadbug,lyingcat")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="action_switch_"))
    out.mkdir(parents=True, exist_ok=True)
    input_recording.RECORDINGS_DIR = out
    torch.set_num_threads(8)
    results = []
    try:
        asyncio.run(scenario(args.sequence.split(","), results))
    except asyncio.CancelledError:
        pass
    print(f"\n=== offline replay of every recording the bridge wrote ({out}) ===", flush=True)
    from production.live_replay import replay_live
    torch.set_grad_enabled(False)
    for path in sorted(out.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        replay = replay_live(path)
        print(replay.summary(), flush=True)
        check(results, header.get("modelSessionFresh") is not None, f"{path.name}: header modelSessionFresh="
              f"{header.get('modelSessionFresh')} action={(header.get('actionState') or {}).get('action')}")
        check(results, replay.parity_ok and (not header.get("modelSessionFresh") or replay.synced.all()),
              f"{path.name}: offline replay from the file alone == live (synced {replay.synced.mean():.0%})")
    failed = [message for ok, message in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
