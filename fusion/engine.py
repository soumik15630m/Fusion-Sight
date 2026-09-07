"""Cross-feed correlation: geolocate this feed's own detections, register
them, and backfill "ghost" detections for objects other feeds saw nearby (in
space and time) that this feed's own model missed.

Called synchronously from detector's per-frame path (detector/app/routers/
track.py, webrtc.py) so ghost markers arrive in the same response as this
feed's own detections -- no separate async budget, per the latency
constraint in multi_feed_fusion_spec.md.
"""
import time

from fusion import config, geo
from fusion.pose_store import Pose
from fusion.registry import registry, matches


def _radius_for(pose: Pose) -> float:
    accuracy = pose.accuracy_m or 0.0
    return config.FUSION_BASE_RADIUS_M + accuracy * config.FUSION_UNCERTAINTY_MULTIPLIER


def _ghost_from(entry, bearing: float, in_frame: bool, pixel) -> dict:
    det = entry.detection
    size = 0.06  # normalised box half-size for an in-frame ghost marker; a placeholder shape, not a measured object extent
    if in_frame and pixel is not None:
        px, py = pixel
        x1, y1, x2, y2 = px - size, py - size, px + size, py + size
    else:
        # Off-screen: client renders an edge indicator from bearing_deg alone,
        # so the box coordinates are unused but kept present/clamped for
        # schema consistency with a normal Detection.
        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

    return {
        "class_id": det.get("class_id", -1),
        "class_name": det.get("class_name", "unknown"),
        "confidence": det.get("confidence", 0.0),
        "x1": max(0.0, min(1.0, x1)),
        "y1": max(0.0, min(1.0, y1)),
        "x2": max(0.0, min(1.0, x2)),
        "y2": max(0.0, min(1.0, y2)),
        "track_id": None,
        "moving": det.get("moving"),
        "is_ghost": True,
        "source_feed": entry.source_id,
        "world_lat": entry.lat,
        "world_lon": entry.lon,
        "bearing_deg": bearing,
        "in_frame": in_frame,
    }


def process(source_id: str, pose: Pose, detections: list[dict]) -> list[dict]:
    """detections: this feed's own Detection dicts for the current frame
    (already built by Detector, un-augmented). Returns detections + ghosts,
    each own detection tagged is_ghost=False."""
    if pose is None:
        return detections

    own_radius = _radius_for(pose)
    own_geo = []

    for det in detections:
        det["is_ghost"] = False
        det.setdefault("source_feed", None)
        det.setdefault("world_lat", None)
        det.setdefault("world_lon", None)
        det.setdefault("bearing_deg", None)
        center = ((det["x1"] + det["x2"]) / 2.0, (det["y1"] + det["y2"]) / 2.0)
        lat, lon = geo.backproject(pose, center)
        det["world_lat"], det["world_lon"] = lat, lon
        own_geo.append((lat, lon))
        registry.register(source_id, lat, lon, own_radius, det)

    ghosts: dict[tuple, dict] = {}  # dedupe key -> ghost, most recent wins
    for entry in registry.others_within_window(source_id):
        # "In proximity" is judged against this feed's own physical position
        # (its camera/pose), not against its detections -- a feed with zero
        # detections this frame must still only backfill objects actually
        # near it, not every other feed's detection anywhere.
        in_proximity = matches(entry.lat, entry.lon, entry.radius_m, pose.lat, pose.lon, own_radius)
        if not in_proximity:
            continue

        already_seen = any(
            matches(entry.lat, entry.lon, entry.radius_m, olat, olon, own_radius)
            for olat, olon in own_geo
        )
        if already_seen:
            continue

        bearing, in_frame, pixel = geo.forward_project(pose, entry.lat, entry.lon)
        key = (entry.source_id, round(entry.lat, 5), round(entry.lon, 5))
        existing = ghosts.get(key)
        if existing is None or entry.ts_ms > existing["_ts_ms"]:
            ghost = _ghost_from(entry, bearing, in_frame, pixel)
            ghost["_ts_ms"] = entry.ts_ms
            ghosts[key] = ghost

    result = list(detections)
    for ghost in ghosts.values():
        ghost.pop("_ts_ms", None)
        result.append(ghost)
    return result


def reset_source(source_id: str) -> None:
    registry.reset_source(source_id)
