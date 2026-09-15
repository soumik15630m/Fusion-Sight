"""Fusion transport endpoints -- pose telemetry ingestion and the live-video
relay for the operator overview page. Both are independent of
/ws/track/{source_id}: pose arrives on its own cadence (fusion/pose_store.py),
and the frame relay just re-publishes bytes /ws/track already decoded
(fusion/frame_relay.py) -- neither touches detector.py or the detection path.
"""
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.schemas import PoseUpdate
from fusion import engine as fusion_engine
from fusion.detection_relay import detection_relay
from fusion.frame_relay import frame_relay
from fusion.pose_store import Pose, pose_store

router = APIRouter(tags=["fusion"])

WS_INTERNAL_ERROR = 1011


@router.websocket("/ws/pose/{source_id}")
async def ws_pose(websocket: WebSocket, source_id: str):
    """Client sends: one JSON PoseUpdate per message, on its own cadence
    (GPS/IMU update rate, not tied to video frame rate)."""
    await websocket.accept()
    print(f"[pose] feed connected: {source_id}")

    try:
        while True:
            raw = await websocket.receive_json()
            try:
                update = PoseUpdate(**raw)
            except ValidationError as e:
                await websocket.send_json({"error": "invalid_pose", "detail": str(e)})
                continue

            pose_store.update(
                source_id,
                Pose(
                    lat=update.lat,
                    lon=update.lon,
                    alt=update.alt,
                    heading_deg=update.heading_deg,
                    tilt_deg=update.tilt_deg,
                    accuracy_m=update.accuracy_m,
                    fov_deg=update.fov_deg,
                    name=update.name,
                ),
            )
    except WebSocketDisconnect:
        print(f"[pose] feed disconnected: {source_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[pose] feed {source_id} failed: {e!r}")
        try:
            await websocket.close(code=WS_INTERNAL_ERROR)
        except RuntimeError:
            pass
    finally:
        pose_store.reset_source(source_id)


@router.websocket("/ws/view/{source_id}")
async def ws_view(websocket: WebSocket, source_id: str):
    """Operator page: subscribes to a feed's already-decoded frames (whatever
    /ws/track or /webrtc/offer for this source_id last published), sent as
    raw JPEG binary messages. Read-only -- a slow/disconnected viewer never
    affects the feed it's watching (bounded queue, oldest frame dropped)."""
    await websocket.accept()
    queue = frame_relay.subscribe(source_id)
    print(f"[view] viewer connected: {source_id}")
    try:
        while True:
            frame = await queue.get()
            await websocket.send_bytes(frame)
    except WebSocketDisconnect:
        print(f"[view] viewer disconnected: {source_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[view] viewer {source_id} failed: {e!r}")
    finally:
        frame_relay.unsubscribe(source_id, queue)


@router.websocket("/ws/detections/{source_id}")
async def ws_detections(websocket: WebSocket, source_id: str):
    """Operator page: the merged detections (own + cross-feed ghosts) for a
    feed, as JSON, on the same cadence its frames arrive on /ws/view. The
    operator video wall overlays these on the relayed video. Read-only, same
    bounded-queue drop-oldest contract as /ws/view -- a slow viewer never
    affects the feed."""
    await websocket.accept()
    queue = detection_relay.subscribe(source_id)
    print(f"[detections] viewer connected: {source_id}")
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        print(f"[detections] viewer disconnected: {source_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[detections] viewer {source_id} failed: {e!r}")
    finally:
        detection_relay.unsubscribe(source_id, queue)


@router.get("/fusion/detections", tags=["fusion"])
async def fusion_detections():
    """Unified world map for the operator overview: every real-world object
    detected across all feeds, deduplicated (an object several feeds see is
    one entry), each with how many feeds confirmed it so the client can
    color-code single-feed sightings differently from cross-confirmed ones."""
    return {"objects": fusion_engine.world_objects()}


@router.get("/fusion/feeds", tags=["fusion"])
async def fusion_feeds():
    """Aggregate view for the map panel (both device and operator pages):
    every feed with a recent pose, plus that pose."""
    active = pose_store.all_active()
    return {
        source_id: {
            "lat": pose.lat,
            "lon": pose.lon,
            "alt": pose.alt,
            "heading_deg": pose.heading_deg,
            "accuracy_m": pose.accuracy_m,
            "name": pose.name,
            "ts_ms": pose.ts_ms,
        }
        for source_id, pose in active.items()
    }
