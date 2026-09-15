"""Operator video relay (detector side). The frames are here -- this streams a
feed's already-decoded JPEGs (whatever /ws/track or /webrtc last published) to
operator viewers. Detection overlays for those frames come separately from the
fusion service (/ws/detections); this endpoint only carries the video.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.frame_relay import frame_relay

router = APIRouter(tags=["streaming"])


@router.websocket("/ws/view/{source_id}")
async def ws_view(websocket: WebSocket, source_id: str):
    """Operator page: subscribe to a feed's frames, sent as raw JPEG binary
    messages. Read-only -- a slow/disconnected viewer never affects the feed
    (bounded queue, oldest frame dropped)."""
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
