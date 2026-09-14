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


def world_objects() -> list[dict]:
    """Collapse every feed's recent geolocated detections into deduplicated
    real-world objects for the operator's unified map. An object seen by 3
    feeds is one marker, not three, and carries how many distinct feeds
    confirmed it (the operator color-codes single-feed vs cross-confirmed).

    Greedy proximity clustering -- linear over the current window's entries,
    which is fine at this project's feed/detection scale (see registry.py's
    module note); it is not a spatial index.
    """
    entries = registry.snapshot_within_window()
    clusters: list[dict] = []

    for e in entries:
        target = None
        for c in clusters:
            if geo.haversine_m(e.lat, e.lon, c["lat"], c["lon"]) <= (e.radius_m + c["radius_m"]):
                target = c
                break
        if target is None:
            clusters.append({
                "lat": e.lat,
                "lon": e.lon,
                "radius_m": e.radius_m,
                "sources": {e.source_id},
                "class_names": [e.detection.get("class_name", "unknown")],
                "count": 1,
            })
        else:
            # Running mean keeps the marker centred on all its members rather
            # than pinned to whichever detection happened to land first.
            n = target["count"]
            target["lat"] = (target["lat"] * n + e.lat) / (n + 1)
            target["lon"] = (target["lon"] * n + e.lon) / (n + 1)
            target["radius_m"] = max(target["radius_m"], e.radius_m)
            target["sources"].add(e.source_id)
            target["class_names"].append(e.detection.get("class_name", "unknown"))
            target["count"] = n + 1

    objects = []
    for c in clusters:
        # Most-reported class label wins as the cluster's name.
        class_name = max(set(c["class_names"]), key=c["class_names"].count)
        objects.append({
            "lat": c["lat"],
            "lon": c["lon"],
            "class_name": class_name,
            "confirmations": len(c["sources"]),
            "sources": sorted(c["sources"]),
        })
    return objects


def reset_source(source_id: str) -> None:
    registry.reset_source(source_id)
