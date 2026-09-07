"""Fan out already-decoded frame bytes to operator-page viewers.

Reuses frames detector's WS/WebRTC handlers already have in memory after
inference -- no second capture/encode on the phone, no SFU/video-relay
server. A viewer (WS /ws/view/{source_id}, detector/app/routers/pose.py)
just drains its own queue; a feed with no viewers pays only the cost of a
`bool(queue)` check per frame.
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
