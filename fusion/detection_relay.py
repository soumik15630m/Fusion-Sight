"""Fan out merged per-frame detections (own + cross-feed ghosts) to operator
viewers.

Parallel to frame_relay (fusion/frame_relay.py): that one relays the raw JPEG
bytes, this one relays the DetectionResponse-shaped JSON the same frame
produced after fusion. The operator video wall needs both -- frames to show,
boxes to overlay -- and a capture device never subscribes here (its own screen
only draws ghosts, which it already has from its /ws/track response).

Same bounded-queue contract as FrameRelay: a slow viewer drops the oldest
payload, never blocks the feed's inference loop.
"""
import asyncio
import threading


class DetectionRelay:
    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, source_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
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

    def publish(self, source_id: str, payload: dict) -> None:
        with self._lock:
            subs = list(self._subscribers.get(source_id, ()))
        for q in subs:
            if q.full():
                try:
                    q.get_nowait()  # drop the oldest queued payload, keep it live rather than laggy
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass


detection_relay = DetectionRelay()
