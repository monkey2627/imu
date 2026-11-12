"""Real-time SolarXR data-feed client for SlimeVR-Server.

Subscribes to SlimeVR-Server's own WebSocket data feed (``ws://127.0.0.1:21110``)
and yields per-tick tracker snapshots: raw fused rotation (``rotation``),
calibrated world-frame rotation (``rotation_reference_adjusted``, the field
SolarXR documents as safe for forward-kinematics reconstruction),
gravity-removed linear acceleration, body-part label and status.

Both rotation fields are needed: SlimeVR-Server computes ``linear_acceleration``
by rotating the tracker's local-frame reading with the RAW rotation, not the
reference-adjusted one (verified against SlimeVR-Server source:
Tracker.kt:456-460 `getAcceleration()` calls
TrackerResetsHandler.kt:196 `getReferenceAdjustedAccel(rawRot, accel) = rawRot.sandwich(accel)`
using `_rotation`/`getRawRotation()`, not `getRotation()`/`getAdjustedRotation()`).
calibration.py uses both rotations together to re-express the acceleration in
the same frame as rotation_reference_adjusted on a per-frame basis.

This speaks the public SolarXR flatbuffers protocol like any other client
(mirrors the handshake in
``IMUTrack-for-Spine/Assets/slimeVR/Scripts/SlimeVrRawImuDataSource.cs``).
Newer local SlimeVR builds may additionally provide
``reference_adjusted_linear_acceleration``. The legacy field remains
available for A/B comparison and backward compatibility.

The generated bindings live in the top-level ``solarxr_protocol`` package
(``E:/dyh/MotionRecover/code/base_mobileposer/solarxr_protocol``), produced by::

    flatc --python --gen-all --gen-object-api -o <out> production/solarxr_schema/all.fbs
"""
from __future__ import annotations

import dataclasses
from typing import AsyncIterator, Dict, Optional, Tuple

import flatbuffers
import websockets

from solarxr_protocol.MessageBundle import MessageBundleT
from solarxr_protocol.data_feed.StartDataFeed import StartDataFeedT
from solarxr_protocol.data_feed.DataFeedConfig import DataFeedConfigT
from solarxr_protocol.data_feed.DataFeedMessage import DataFeedMessage
from solarxr_protocol.data_feed.DataFeedMessageHeader import DataFeedMessageHeaderT
from solarxr_protocol.data_feed.device_data.DeviceDataMask import DeviceDataMaskT
from solarxr_protocol.data_feed.tracker.TrackerDataMask import TrackerDataMaskT
from solarxr_protocol.datatypes.BodyPart import BodyPart
from solarxr_protocol.datatypes.TrackerStatus import TrackerStatus
from solarxr_protocol.rpc.ResetRequest import ResetRequestT
from solarxr_protocol.rpc.ResetType import ResetType
from solarxr_protocol.rpc.RpcMessage import RpcMessage
from solarxr_protocol.rpc.RpcMessageHeader import RpcMessageHeaderT

ENDPOINT = "ws://127.0.0.1:21110"


@dataclasses.dataclass(frozen=True)
class TrackerSample:
    """One tracker's data from a single DataFeedUpdate tick."""

    tracker_key: str          # "{device_id}:{tracker_num}", stable per physical tracker
    device_id: int
    tracker_num: int
    body_part: int            # BodyPart enum value (see solarxr_protocol.datatypes.BodyPart)
    status: int                # TrackerStatus enum value
    raw_quat_xyzw: Tuple[float, float, float, float]        # rotation (un-adjusted fusion output)
    quat_xyzw: Tuple[float, float, float, float]             # rotation_reference_adjusted, world frame
    accel_xyz: Tuple[float, float, float]         # linear_acceleration, m/s^2, gravity removed,
                                                    # in the SAME (raw-rotation) frame as raw_quat_xyzw
    reference_adjusted_accel_xyz: Optional[Tuple[float, float, float]] = None
    raw_angular_velocity_xyz: Optional[Tuple[float, float, float]] = None
    raw_acceleration_xyz: Optional[Tuple[float, float, float]] = None

    @property
    def online(self) -> bool:
        return self.status in (TrackerStatus.OK, TrackerStatus.BUSY)

    @property
    def body_part_name(self) -> str:
        return _BODY_PART_NAMES.get(self.body_part, f"UNKNOWN({self.body_part})")


_BODY_PART_NAMES = {
    value: name
    for name, value in vars(BodyPart).items()
    if not name.startswith("_") and isinstance(value, int)
}


def _build_start_feed_request(minimum_time_since_last_ms: int) -> bytes:
    tracker_mask = TrackerDataMaskT(
        info=True,
        status=True,
        rotation=True,
        rotationReferenceAdjusted=True,
        linearAcceleration=True,
        rawAngularVelocity=True,
        rawAcceleration=True,
        referenceAdjustedLinearAcceleration=True,
    )
    device_mask = DeviceDataMaskT(trackerData=tracker_mask, deviceData=True)
    config = DataFeedConfigT(
        minimumTimeSinceLast=minimum_time_since_last_ms,
        dataMask=device_mask,
    )
    start = StartDataFeedT(dataFeeds=[config])
    header = DataFeedMessageHeaderT(messageType=DataFeedMessage.StartDataFeed, message=start)
    bundle = MessageBundleT(dataFeedMsgs=[header])

    builder = flatbuffers.Builder(1024)
    offset = bundle.Pack(builder)
    builder.Finish(offset)
    return bytes(builder.Output())


async def request_reset(reset_type: int = ResetType.Yaw, endpoint: str = ENDPOINT) -> None:
    """Ask SlimeVR-Server for a reset (Yaw / Full / Mounting) over a short-lived SolarXR connection."""
    header = RpcMessageHeaderT(messageType=RpcMessage.ResetRequest, message=ResetRequestT(resetType=reset_type))
    builder = flatbuffers.Builder(256)
    builder.Finish(MessageBundleT(rpcMsgs=[header]).Pack(builder))
    async with websockets.connect(endpoint, proxy=None, ping_interval=None, close_timeout=2) as websocket:
        await websocket.send(bytes(builder.Output()))


def _parse_bundle(raw: bytes) -> Dict[str, TrackerSample]:
    """Parse one MessageBundle into {tracker_key: TrackerSample}.

    Only trackers with both rotation_reference_adjusted and linear_acceleration
    populated are included -- callers should not have to null-check every field.
    """
    samples: Dict[str, TrackerSample] = {}
    # InitFromBuf expects an already-resolved root-table position; a raw
    # buffer straight off the wire (or from builder.Output()) still has the
    # leading indirect uoffset, so it must go through InitFromPackedBuf.
    bundle = MessageBundleT.InitFromPackedBuf(raw, 0)
    if not bundle.dataFeedMsgs:
        return samples

    for header in bundle.dataFeedMsgs:
        if header is None or header.messageType != DataFeedMessage.DataFeedUpdate:
            continue
        update = header.message
        if update is None or not update.devices:
            continue
        for device in update.devices:
            if device is None or device.id is None or not device.trackers:
                continue
            for tracker in device.trackers:
                if tracker is None or tracker.trackerId is None:
                    continue
                device_id = (
                    tracker.trackerId.deviceId.id
                    if tracker.trackerId.deviceId is not None
                    else device.id.id
                )
                tracker_num = tracker.trackerId.trackerNum
                raw_quat = tracker.rotation
                quat = tracker.rotationReferenceAdjusted
                accel = tracker.linearAcceleration
                raw_gyro = getattr(tracker, "rawAngularVelocity", None)
                raw_accel = getattr(tracker, "rawAcceleration", None)
                reference_accel = getattr(tracker, "referenceAdjustedLinearAcceleration", None)
                if raw_quat is None or quat is None or accel is None:
                    continue
                key = f"{device_id}:{tracker_num}"
                samples[key] = TrackerSample(
                    tracker_key=key,
                    device_id=device_id,
                    tracker_num=tracker_num,
                    body_part=tracker.info.bodyPart if tracker.info is not None else BodyPart.NONE,
                    status=tracker.status,
                    raw_quat_xyzw=(raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w),
                    quat_xyzw=(quat.x, quat.y, quat.z, quat.w),
                    accel_xyz=(accel.x, accel.y, accel.z),
                    reference_adjusted_accel_xyz=(
                        (reference_accel.x, reference_accel.y, reference_accel.z)
                        if reference_accel is not None else None
                    ),
                    raw_angular_velocity_xyz=(
                        (raw_gyro.x, raw_gyro.y, raw_gyro.z)
                        if raw_gyro is not None else None
                    ),
                    raw_acceleration_xyz=(
                        (raw_accel.x, raw_accel.y, raw_accel.z)
                        if raw_accel is not None else None
                    ),
                )
    return samples


async def stream_trackers(
    endpoint: str = ENDPOINT,
    minimum_time_since_last_ms: int = 33,
) -> AsyncIterator[Dict[str, TrackerSample]]:
    """Yield a merged {tracker_key: TrackerSample} dict on every DataFeedUpdate.

    Each yielded dict only contains trackers that appeared in that particular
    update (SolarXR only sends deltas for what changed), so callers that need
    a full current snapshot should keep merging into their own dict keyed by
    tracker_key, the same way SlimeVrRawImuDataSource.cs does on the Unity side.
    """
    request = _build_start_feed_request(minimum_time_since_last_ms)
    # Keep the SolarXR socket alive while the server is quiet during startup
    # calibration.  The default 20s timeout is too aggressive for some
    # SlimeVR builds and presents as "no close frame received or sent".
    async with websockets.connect(
        endpoint,
        max_size=None,
        # SolarXR is local.  Do not let websockets 16 route it through a
        # system HTTP proxy, and avoid ping frames that older SolarXR builds
        # may not answer consistently.
        proxy=None,
        ping_interval=None,
        ping_timeout=None,
        close_timeout=5,
    ) as ws:
        await ws.send(request)
        async for message in ws:
            if not isinstance(message, (bytes, bytearray)):
                continue
            try:
                samples = _parse_bundle(bytes(message))
            except Exception:
                # Malformed/partial frame; drop it and keep the connection alive.
                continue
            if samples:
                yield samples
