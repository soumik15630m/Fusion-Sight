import json

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.detector import detector
from fusion import config as fusion_config
from fusion import engine as fusion_engine
from fusion.detection_relay import detection_relay
from fusion.frame_relay import frame_relay
from fusion.pose_store import pose_store

router = APIRouter(tags=["streaming"])

# Close codes (RFC 6455): 1011 = server hit an unexpected condition.
WS_INTERNAL_ERROR = 1011


@router.websocket("/ws/track/{source_id}")
async def ws_track(websocket: WebSocket, source_id: str, view: str = "ground"):
    """Live feed endpoint.

    Client sends: raw JPEG bytes, one binary message per frame.
    Server sends: one JSON DetectionResponse per frame.

    `view` is a query param on the connect URL (e.g. `?view=drone`), fixed for
    the life of the connection -- one feed is one camera angle, it doesn't
    switch mid-stream. 'ground' (default) or 'drone', see app/config.py's
    DRONE_MODEL_PATH.
    """
    await websocket.accept()

    if detector.model is None:
        # Without this the first frame dies on an AttributeError inside the
        # detector and the client just sees the socket drop.
        await websocket.send_text(json.dumps({"error": "model_not_loaded"}))
        await websocket.close(code=WS_INTERNAL_ERROR)
        return

    print(f"[ws] feed connected: {source_id}")

    try:
        while True:
            raw = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({"error": "decode_failed"}))
                continue

            # Blocking inference belongs off the event loop: otherwise this feed
            # blocks every other feed and every HTTP request for its duration.
            result = await run_in_threadpool(detector.track, frame, source_id, view)

            # Fusion runs synchronously in the same per-frame path (not a
            # separate async budget) -- see multi_feed_fusion_spec.md's
            # latency decision. No-op if fusion is off or this feed has no
            # pose yet (own detections pass through unchanged).
            if fusion_config.FUSION_ENABLED:
                pose = pose_store.latest(source_id)
                if pose is not None:
                    result["detections"] = fusion_engine.process(
                        source_id, pose, result["detections"]
                    )

            await websocket.send_text(json.dumps(result))
            # Fan the frame out to operator viewers, and the merged detections
            # alongside it so the operator video wall can overlay boxes on the
            # same frame -- the capture device draws only ghosts from its own
            # response, the merge lives on the operator side.
            frame_relay.publish(source_id, raw)
            detection_relay.publish(source_id, result)

    except WebSocketDisconnect:
        print(f"[ws] feed disconnected: {source_id}")
    except Exception as e:  # noqa: BLE001
        # A decode/inference failure must not leak this feed's tracker state.
        print(f"[ws] feed {source_id} failed: {e!r}")
        try:
            await websocket.close(code=WS_INTERNAL_ERROR)
        except RuntimeError:
            pass  # already closed
    finally:
        # Runs on every exit path, so a crashed feed frees its state too.
        detector.reset_source(source_id)
        fusion_engine.reset_source(source_id)
