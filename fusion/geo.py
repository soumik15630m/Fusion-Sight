"""Pixel <-> world-coordinate projection.

Plain spherical-earth trigonometry (haversine / great-circle destination
point), not a geodesy library (pyproj/utm) -- at the tens-to-hundreds-of-
meters scale this fusion operates at, the sphere approximation's error is
negligible next to the GPS accuracy it's already working around, and it
avoids pulling in a new dependency (and its own install headaches -- see
detector/README.md's TensorRT section for what that can look like) for a
demo-scope feature.

Ground-plane assumption: a detection sits on flat ground at the camera's own
altitude (pose.alt). That's the same simplifying assumption the aerial
ego-compensation path already makes elsewhere in this project (see
detector/README.md "Ego compensation fails under parallax") -- valid for a
distant, roughly level scene, not for significant elevation change between
camera and target. Known limitation, not solved here.
"""
import math
from typing import Optional

from fusion import config
from fusion.pose_store import Pose

EARTH_RADIUS_M = 6371000.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination_point(lat: float, lon: float, bearing: float, distance_m: float) -> tuple[float, float]:
    """Given a start point, bearing (deg) and distance (m), the resulting
    lat/lon along that great circle."""
    p1 = math.radians(lat)
    brng = math.radians(bearing)
    d_r = distance_m / EARTH_RADIUS_M
    p2 = math.asin(math.sin(p1) * math.cos(d_r) + math.cos(p1) * math.sin(d_r) * math.cos(brng))
    l2 = math.radians(lon) + math.atan2(
        math.sin(brng) * math.sin(d_r) * math.cos(p1),
        math.cos(d_r) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(p2), math.degrees(l2)


def _fov(pose: Pose) -> float:
    return pose.fov_deg or config.FUSION_DEFAULT_FOV_DEG


def backproject(
    pose: Pose, bbox_center_norm: tuple[float, float], fov_deg: Optional[float] = None
) -> tuple[float, float]:
    """Project a normalised (0-1) pixel center to a real-world (lat, lon),
    assuming it sits on the ground plane at the camera's own altitude.

    Horizontal: offset from frame center scaled by FOV gives a bearing
    relative to the camera's heading.
    Vertical: offset from frame center scaled by (assumed) vertical FOV gives
    a depression angle below the camera's tilt; ground distance follows from
    simple right-triangle trig against the camera's altitude. If the
    resulting line of sight doesn't point below the horizon at all (looking
    at/above it), there's no real ground intersection -- fall back to a fixed
    assumed distance along the same bearing rather than dropping the
    detection (see FUSION_FALLBACK_DISTANCE_M).
    """
    fov = fov_deg or _fov(pose)
    cx, cy = bbox_center_norm
    h_fov = fov
    v_fov = fov * 0.75  # typical 4:3-ish vertical FOV fraction of horizontal; approximate, not measured per-device

    bearing = (pose.heading_deg + (cx - 0.5) * h_fov) % 360
    depression = pose.tilt_deg + (0.5 - cy) * v_fov  # deg below horizontal, positive = down

    if depression <= 1.0:  # near/above horizon -- degenerate intersection
        distance = config.FUSION_FALLBACK_DISTANCE_M
    else:
        distance = pose.alt / math.tan(math.radians(depression)) if pose.alt > 0 else config.FUSION_FALLBACK_DISTANCE_M
        distance = _clamp(distance, 0.5, 2000.0)

    return destination_point(pose.lat, pose.lon, bearing, distance)


def forward_project(
    pose: Pose, lat: float, lon: float, fov_deg: Optional[float] = None
) -> tuple[float, bool, Optional[tuple[float, float]]]:
    """Inverse of backproject: given this feed's pose and a world point,
    return (bearing_deg from this feed to the point, whether it falls inside
    this feed's current frame, and a normalised pixel (x, y) if so).

    Vertical placement is not modelled (no target-height information to
    invert) -- an in-frame ghost is placed at the frame's vertical center;
    off-screen ghosts rely on bearing_deg alone (client draws an edge
    indicator), which is the case that actually matters for the AR use case.
    """
    fov = fov_deg or _fov(pose)
    brg = bearing_deg(pose.lat, pose.lon, lat, lon)
    rel = ((brg - pose.heading_deg + 180) % 360) - 180  # signed, [-180, 180]

    in_frame = abs(rel) <= fov / 2
    pixel = None
    if in_frame:
        px = _clamp(0.5 + rel / fov, 0.0, 1.0)
        pixel = (px, 0.5)

    return brg, in_frame, pixel
