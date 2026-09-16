import json

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.detector import detector
from app.frame_relay import frame_relay
from app.fusion_client import fusion_client

router = APIRouter(tags=["streaming"])

# Close codes (RFC 6455): 1011 = server hit an unexpected condition.
WS_INTERNAL_ERROR = 1011


@router.websocket("/ws/track/{source_id}")
async def ws_track(websocket: WebSocket, source_id: str, view: str = "ground"):
    """Live feed endpoint (detector side only -- inference, no fusion).

    Client sends: raw JPEG bytes, one binary message per frame.
    Server sends: one JSON DetectionResponse (this feed's OWN detections) per
    frame -- the phone's screen shows only ghosts, which arrive separately from
    the fusion service, so this response is really for status/debug.

    Each frame also (a) gets republished to operator viewers via the frame relay
    (/ws/view) and (b) has its raw detections streamed to the fusion service,
    which pairs them with pose and computes the cross-feed ghosts.

    `view` is a query param fixed for the connection ('ground' default, or
    'drone'); one feed is one camera angle.
    """
    await websocket.accept()

    if detector.model is None:
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

            # Blocking inference off the event loop so one feed can't stall the others.
            result = await run_in_threadpool(detector.track, frame, source_id, view)
            own = result.get("detections", [])

            # The phone's screen shows only ghosts (from fusion), so returning the
            # full detection list here is wasted downlink -- send a compact status
            # ack instead, and reserve the full detections for fusion + operator.
            await websocket.send_text(
                json.dumps(
                    {
                        "source_id": source_id,
                        "inference_ms": result.get("inference_ms", 0.0),
                        "own_count": len(own),
                        "frame_width": result.get("frame_width", 0),
                        "frame_height": result.get("frame_height", 0),
                    }
                )
            )
            frame_relay.publish(source_id, raw)
            # Hand the full raw detections to the fusion service (non-blocking).
            fusion_client.send(
                {
                    "source_id": source_id,
                    "detections": own,
                    "frame_width": result.get("frame_width", 0),
                    "frame_height": result.get("frame_height", 0),
                    "inference_ms": result.get("inference_ms", 0.0),
                }
            )

    except WebSocketDisconnect:
        print(f"[ws] feed disconnected: {source_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[ws] feed {source_id} failed: {e!r}")
        try:
            await websocket.close(code=WS_INTERNAL_ERROR)
        except RuntimeError:
            pass  # already closed
    finally:
        # Free this feed's tracker state, and tell fusion to drop its pose/registry.
        detector.reset_source(source_id)
        fusion_client.send({"source_id": source_id, "event": "end"})
