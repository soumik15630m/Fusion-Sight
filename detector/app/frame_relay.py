"""Fan out already-decoded frame bytes to operator-page viewers.

Lives with the detector because the frames are here: /ws/track and
/webrtc/offer already have the JPEG in hand after inference, and /ws/view
(app/routers/view.py) re-publishes those same bytes -- no second capture/encode
on the phone, and frames never cross to the fusion service (only the light
detection JSON does). A feed with no viewers pays only a `bool(queue)` check
per frame.
"""
import asyncio
import threading


class FrameRelay:
    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, source_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)  # bounded: a slow viewer drops frames, never blocks the feed
        with self._lock:
            self._subscribers.setdefault(source_id, set()).add(q)
        return q

    def unsubscribe(self, source_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(source_id)
            if subs:
                subs.discard(q)
                if not subs:
                    self._subscribers.pop(source_id, None)

    def publish(self, source_id: str, jpeg_bytes: bytes) -> None:
        with self._lock:
            subs = list(self._subscribers.get(source_id, ()))
        for q in subs:
            if q.full():
                try:
                    q.get_nowait()  # drop the oldest queued frame, keep it live rather than laggy
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(jpeg_bytes)
            except asyncio.QueueFull:
                pass


frame_relay = FrameRelay()
