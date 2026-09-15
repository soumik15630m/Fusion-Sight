"""Outbound link from the detector to the fusion service.

The detector runs inference and streams the raw detections here; this holds one
persistent WebSocket to the fusion laptop's /ws/ingest and drains a bounded
queue onto it. `send()` is non-blocking and drops the oldest payload if the
link is slow, so a fusion hiccup can never stall the GPU inference loop -- the
phone still gets its own detections, only the cross-feed ghosts pause.

FUSION_WS_URL points at the fusion service (over the private mesh in the
distributed deployment, or a compose service name on one machine).
"""
import asyncio
import json
import os

import websockets

FUSION_WS_URL = os.getenv("FUSION_WS_URL", "ws://fusion:8100/ws/ingest")
_RECONNECT_S = 2.0


class FusionClient:
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._task: asyncio.Task | None = None

    def send(self, payload: dict) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()  # drop oldest, keep the link live not backed-up
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    async def _run(self) -> None:
        while True:
            try:
                async with websockets.connect(FUSION_WS_URL, max_queue=None) as ws:
                    print(f"[fusion-client] connected to {FUSION_WS_URL}")
                    while True:
                        payload = await self._queue.get()
                        await ws.send(json.dumps(payload))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"[fusion-client] link down ({e!r}); retrying in {_RECONNECT_S}s")
                await asyncio.sleep(_RECONNECT_S)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


fusion_client = FusionClient()
