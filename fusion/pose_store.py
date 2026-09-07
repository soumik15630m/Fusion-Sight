"""Per-feed pose (GPS + IMU) history.

Pose arrives over its own channel (WS /ws/pose/{source_id} in
detector/app/routers/pose.py), independent of the video/detection path -- a
feed's pose update rate has nothing to do with its frame rate.
"""
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from fusion import config


@dataclass
class Pose:
    lat: float
    lon: float
    alt: float = 0.0
    heading_deg: float = 0.0     # compass heading, 0 = north, clockwise
    tilt_deg: float = 0.0        # camera pitch below horizontal; positive = looking down
    accuracy_m: Optional[float] = None
    fov_deg: Optional[float] = None
    ts_ms: float = 0.0           # server receipt time, set by PoseStore.update


class PoseStore:
    """Thread-safe, mirrors Detector's per-source_id dict + single lock pattern
    (app/detector.py) rather than inventing a new concurrency model."""

    def __init__(self):
        self._lock = threading.Lock()
        self._poses: dict[str, deque[Pose]] = {}

    def update(self, source_id: str, pose: Pose) -> None:
        pose.ts_ms = time.time() * 1000
        with self._lock:
            history = self._poses.setdefault(
                source_id, deque(maxlen=config.FUSION_POSE_HISTORY)
            )
            history.append(pose)

    def latest(self, source_id: str) -> Optional[Pose]:
        with self._lock:
            history = self._poses.get(source_id)
            if not history:
                return None
            pose = history[-1]
        if time.time() * 1000 - pose.ts_ms > config.FUSION_POSE_MAX_AGE_MS:
            return None
        return pose

    def all_active(self, max_age_ms: Optional[int] = None) -> dict[str, Pose]:
        max_age_ms = config.FUSION_POSE_MAX_AGE_MS if max_age_ms is None else max_age_ms
        now = time.time() * 1000
        with self._lock:
            snapshot = {sid: h[-1] for sid, h in self._poses.items() if h}
        return {sid: p for sid, p in snapshot.items() if now - p.ts_ms <= max_age_ms}

    def reset_source(self, source_id: str) -> None:
        with self._lock:
            self._poses.pop(source_id, None)


pose_store = PoseStore()
