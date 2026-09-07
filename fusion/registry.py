"""Recent geolocated detections across all feeds, for cross-feed proximity
matching. Linear scan is fine at N=4-5 feeds a handful of detections each per
sliding window; a spatial index is a non-goal until this scales well past a
handful of concurrent feeds.
"""
import threading
import time
from dataclasses import dataclass

from fusion import config, geo


@dataclass
class RegistryEntry:
    source_id: str
    lat: float
    lon: float
    radius_m: float
    detection: dict
    ts_ms: float


class DetectionRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries: list[RegistryEntry] = []

    def register(self, source_id: str, lat: float, lon: float, radius_m: float, detection: dict) -> None:
        entry = RegistryEntry(source_id, lat, lon, radius_m, detection, time.time() * 1000)
        with self._lock:
            self._entries.append(entry)

    def _prune_locked(self, now_ms: float) -> None:
        cutoff = now_ms - config.FUSION_WINDOW_MS
        self._entries = [e for e in self._entries if e.ts_ms >= cutoff]

    def others_within_window(self, exclude_source_id: str) -> list[RegistryEntry]:
        now = time.time() * 1000
        with self._lock:
            self._prune_locked(now)
            return [e for e in self._entries if e.source_id != exclude_source_id]

    def reset_source(self, source_id: str) -> None:
        with self._lock:
            self._entries = [e for e in self._entries if e.source_id != source_id]


def matches(a_lat: float, a_lon: float, a_radius: float, b_lat: float, b_lon: float, b_radius: float) -> bool:
    return geo.haversine_m(a_lat, a_lon, b_lat, b_lon) <= (a_radius + b_radius)


registry = DetectionRegistry()
