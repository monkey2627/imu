"""Coordinate Android video capture with a SolarXR research session."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from aiohttp import web

from production.capture import SessionWriter, _json
from runtime_contract import REALTIME_LAYOUT
from production.layout import LayoutMismatchError, detect_layout
from production.solarxr_client import stream_trackers


def start_message(session_id: str, upload_url: str, duration_seconds: int) -> dict:
    return {"type": "start_recording", "session_id": session_id,
            "server_time_unix_ms": int(time.time() * 1000),
            "duration_seconds": duration_seconds, "upload_url": upload_url}


def stop_message(session_id: str) -> dict:
    return {"type": "stop_recording", "session_id": session_id}


class CaptureServer:
    def __init__(self, args):
        self.args = args
        self.phone = None
        self.phone_connected = asyncio.Event()
        self.messages = asyncio.Queue()
        self.sessions: dict[str, dict] = {}
        self.active_session = None

    async def websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.phone = ws
        self.phone_connected.set()
        async for message in ws:
            if message.type == web.WSMsgType.TEXT:
                try:
                    payload = json.loads(message.data)
                except json.JSONDecodeError:
                    continue
                await self.messages.put(payload)
        self.phone = None
        self.phone_connected.clear()
        return ws

    async def health(self, _request):
        return web.json_response({"ok": True, "phone_connected": self.phone is not None,
                                  "active_session": self.active_session})

    async def session(self, request):
        record = self.sessions.get(request.match_info["session_id"])
        if record is None:
            raise web.HTTPNotFound()
        return web.json_response(record)

    async def upload(self, request):
        if not request.content_type.startswith("multipart/"):
            fields = await request.post()
            session_id = fields.get("session_id") or request.query.get("session_id")
            if not session_id or session_id != self.active_session:
                raise web.HTTPConflict(text="unknown or inactive session_id")
            raise web.HTTPBadRequest(text="multipart field 'file' is required")

        reader = await request.multipart()
        fields = {}
        video_part = None
        while part := await reader.next():
            if part.name == "file":
                video_part = part
                break
            fields[part.name] = await part.text()
        session_id = fields.get("session_id") or request.query.get("session_id")
        if not session_id or session_id != self.active_session:
            raise web.HTTPConflict(text="unknown or inactive session_id")
        if video_part is None:
            raise web.HTTPBadRequest(text="multipart field 'file' is required")
        target = self.args.output / session_id / "phone_video.mp4"
        digest = hashlib.sha256()
        size = 0
        with target.open("wb") as output:
            while chunk := await video_part.read_chunk(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        metadata = {"session_id": session_id, "file_name": target.name,
                    "size_bytes": size, "sha256": digest.hexdigest(),
                    "received_unix_ms": int(time.time() * 1000),
                    "phone_metadata": fields}
        (target.parent / "video_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.sessions[session_id].update({"video": metadata, "upload_complete": True})
        if self.phone and not self.phone.closed:
            await self.phone.send_json({"type": "upload_received", **metadata})
        return web.json_response(metadata)

    def app(self):
        app = web.Application(client_max_size=20 * 1024**3)
        app.router.add_get("/control", self.websocket)
        app.router.add_get("/health", self.health)
        app.router.add_get("/sessions/{session_id}", self.session)
        app.router.add_post("/upload", self.upload)
        return app

    async def receive_type(self, kind, session_id, timeout=15):
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {kind}")
            message = await asyncio.wait_for(self.messages.get(), remaining)
            if message.get("session_id") != session_id:
                continue
            if message.get("type") == "error":
                raise RuntimeError(f"Phone error {message.get('code')}: {message.get('message')}")
            if message.get("type") == kind:
                return message

    async def run_session(self):
        print("Waiting for Android phone at /control...", flush=True)
        await self.phone_connected.wait()
        source = stream_trackers(minimum_time_since_last_ms=self.args.interval_ms).__aiter__()
        latest, count = {}, 0
        first_sample_unix_ms = 0
        first_sample_monotonic_ns = 0
        while True:
            first = await source.__anext__()
            first_sample_unix_ms = int(time.time() * 1000)
            first_sample_monotonic_ns = time.monotonic_ns()
            latest.update(first)
            try:
                layout = detect_layout(latest)
                break
            except LayoutMismatchError:
                continue
        if self.args.layout != "auto" and layout.name != self.args.layout:
            raise RuntimeError(f"Expected {self.args.layout}, got {layout.name}")
        session_id = f"{self.args.subject_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        root = self.args.output / session_id
        self.args.output.mkdir(parents=True, exist_ok=True)
        writer = SessionWriter(root, self.args)
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"sessionId": session_id, "captureMode": "android_video_plus_solarxr",
                         "solarxrEndpoint": "ws://127.0.0.1:21110"})
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.active_session = session_id
        self.sessions[session_id] = {"session_id": session_id, "status": "recording"}
        writer.event("layout_resolved", layout=layout.name, labels=layout.labels,
                     trackerKeys=layout.label_to_tracker_key)
        for update in (first,):
            count += 1
            writer.update(count, update)
        upload_url = f"http://{self.args.advertise_host}:{self.args.port}/upload"
        command = start_message(session_id, upload_url, self.args.duration_seconds)
        command["server_time_unix_ms"] = int(time.time() * 1000)
        writer.event("start_command_sent", commandSendUnixMs=command["server_time_unix_ms"],
                     hostMonotonicNs=time.monotonic_ns())
        try:
            await self.phone.send_json(command)
            ack = await self.receive_type("ack", session_id)
            writer.event("phone_ack", **ack)
            started = await self.receive_type("recording_started", session_id)
            writer.event("phone_recording_started", **started, syncMarker="phone_beep_and_operator_clap_or_stomp")
            writer.event("session_started", solarxrFirstSampleHostUnixMs=first_sample_unix_ms,
                         solarxrFirstSampleHostMonotonicNs=first_sample_monotonic_ns)
            deadline = asyncio.get_running_loop().time() + self.args.duration_seconds
            while asyncio.get_running_loop().time() < deadline:
                update = await source.__anext__()
                latest.update(update)
                count += 1
                writer.update(count, update)
            await self.phone.send_json(stop_message(session_id))
            writer.event("stop_command_sent", commandSendUnixMs=int(time.time()*1000),
                         hostMonotonicNs=time.monotonic_ns())
            stopped = await self.receive_type("recording_stopped", session_id, timeout=60)
            writer.event("phone_recording_stopped", **stopped)
            self.sessions[session_id]["status"] = "awaiting_upload"
            self.sessions[session_id]["phone"] = stopped
            writer.event("session_ended", totalUpdates=count)
        except BaseException as exc:
            writer.event("session_aborted", error=f"{type(exc).__name__}: {exc}", totalUpdates=count)
            self.sessions[session_id]["status"] = "error"
            raise
        finally:
            writer.close()


async def serve(args):
    service = CaptureServer(args)
    runner = web.AppRunner(service.app())
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    print(f"Capture control listening on {args.host}:{args.port}", flush=True)
    try:
        await service.run_session()
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--advertise-host", default="127.0.0.1", help="LAN IP reachable by phone")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output", type=Path, default=Path("recordings/research"))
    parser.add_argument("--subject-id", default="pilot01")
    parser.add_argument("--layout", default=REALTIME_LAYOUT)
    parser.add_argument("--duration-seconds", type=int, default=120)
    parser.add_argument("--interval-ms", type=int, default=16)
    parser.add_argument("--validation-seconds", type=float, default=0)
    parser.add_argument("--video-path", default="")
    parser.add_argument("--reference-path", default="")
    parser.add_argument("--notes", default="Android coordinated capture")
    args = parser.parse_args()
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
