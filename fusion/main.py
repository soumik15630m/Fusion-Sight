"""Standalone fusion service (its own laptop).

Owns all cross-feed state -- pose_store, the detection registry, and the two
fan-out relays -- and does the geolocation correlation. It never touches the
GPU: the detector (a separate service) runs inference and streams raw
detections here over /ws/ingest; this service pairs them with poses, computes
"ghost" backfills, and pushes results out:

  phones  --/ws/pose/{id}-->      pose in
  detector --/ws/ingest-->        raw detections in (tagged with source_id)
  phones  <--/ws/ghosts/{id}--    ghost markers for the wearer's AR screen
  operator<--/ws/detections/{id}--merged own+ghost boxes for the video wall
  operator  GET /fusion/feeds     live feed positions (+ names) for the map
  operator  GET /fusion/detections deduped world objects for the map

The video frames themselves stay on the detector (it serves /ws/view) -- only
the lightweight detection JSON crosses to here, so frames are never shipped
twice.
"""
import json

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from fusion import config, engine
from fusion.detection_relay import DetectionRelay, detection_relay
from fusion.pose_store import Pose, pose_store
from fusion.schemas import PoseUpdate

# Ghosts to phones travel on their own relay (the same bounded drop-oldest
# fan-out as the operator's detection relay), keyed by source_id.
ghost_relay = DetectionRelay()

WS_INTERNAL_ERROR = 1011

router = APIRouter()


@router.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "service": "fusion",
        "fusion_enabled": config.FUSION_ENABLED,
        "base_radius_m": config.FUSION_BASE_RADIUS_M,
        "window_ms": config.FUSION_WINDOW_MS,
        "feeds_with_pose": list(pose_store.all_active().keys()),
    }


@router.websocket("/ws/pose/{source_id}")
async def ws_pose(websocket: WebSocket, source_id: str):
    """Phone -> fusion: one JSON PoseUpdate per message, on the GPS/IMU cadence."""
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


@router.websocket("/ws/ingest")
async def ws_ingest(websocket: WebSocket):
    """Detector -> fusion: raw per-frame detections, one JSON message per frame,
    each tagged with its source_id (one shared connection carries all feeds).

    Message shape:
      {"source_id","detections":[...],"frame_width","frame_height","inference_ms"}
    or a lifecycle marker when a feed goes away:
      {"source_id","event":"end"}

    Fusion pairs the detections with that feed's latest pose, computes ghosts,
    and fans the results out to the phone (ghosts) and operator (everything)."""
    await websocket.accept()
    print("[ingest] detector connected")
    try:
        while True:
            msg = await websocket.receive_json()
            source_id = msg.get("source_id")
            if not source_id:
                continue

            if msg.get("event") == "end":
                engine.reset_source(source_id)
                pose_store.reset_source(source_id)
                continue

            detections = msg.get("detections", [])
            pose = pose_store.latest(source_id) if config.FUSION_ENABLED else None
            if pose is not None:
                merged = engine.process(source_id, pose, detections)
            else:
                # No pose yet (or fusion off): own detections pass through, no ghosts.
                for d in detections:
                    d["is_ghost"] = False
                merged = detections

            payload = {
                "source_id": source_id,
                "frame_width": msg.get("frame_width", 0),
                "frame_height": msg.get("frame_height", 0),
                "inference_ms": msg.get("inference_ms", 0.0),
                "detections": merged,
            }
            detection_relay.publish(source_id, payload)  # operator: all boxes

            ghosts = [d for d in merged if d.get("is_ghost")]
            ghost_relay.publish(source_id, {**payload, "detections": ghosts})  # phone: ghosts only
    except WebSocketDisconnect:
        print("[ingest] detector disconnected")
    except Exception as e:  # noqa: BLE001
        print(f"[ingest] failed: {e!r}")
        try:
            await websocket.close(code=WS_INTERNAL_ERROR)
        except RuntimeError:
            pass


@router.websocket("/ws/ghosts/{source_id}")
async def ws_ghosts(websocket: WebSocket, source_id: str):
    """Phone AR screen: the cross-feed ghost markers for this feed."""
    await websocket.accept()
    queue = ghost_relay.subscribe(source_id)
    print(f"[ghosts] wearer connected: {source_id}")
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        print(f"[ghosts] wearer disconnected: {source_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[ghosts] {source_id} failed: {e!r}")
    finally:
        ghost_relay.unsubscribe(source_id, queue)


@router.websocket("/ws/detections/{source_id}")
async def ws_detections(websocket: WebSocket, source_id: str):
    """Operator video wall: the merged own+ghost detections for a feed."""
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
        print(f"[detections] {source_id} failed: {e!r}")
    finally:
        detection_relay.unsubscribe(source_id, queue)


@router.get("/fusion/feeds", tags=["fusion"])
async def fusion_feeds():
    """Every feed with a recent pose, plus that pose -- for the operator map."""
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


@router.get("/fusion/detections", tags=["fusion"])
async def fusion_detections():
    """Deduplicated real-world objects across all feeds, with a confirmations
    count, for the operator's unified map."""
    return {"objects": engine.world_objects()}


app = FastAPI(title="FusionSight - Fusion Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.include_router(router)
