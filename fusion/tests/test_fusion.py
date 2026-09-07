"""Fusion engine tests -- no network, no detector, no model. Run from repo
root: `python -m pytest fusion/tests`."""
import math
import time

import pytest

from fusion import engine, geo
from fusion.pose_store import Pose
from fusion.registry import registry


@pytest.fixture(autouse=True)
def _clean_registry():
    # engine.process shares a module-level registry across tests/feeds --
    # same reason app/detector.py's tests reset per-source_id state (see
    # detector/tests/test_feed_isolation.py).
    yield
    for sid in ("feed_a", "feed_b", "feed_c"):
        registry.reset_source(sid)


def _det(class_name="personnel", conf=0.8, cx=0.5, cy=0.5):
    half = 0.02
    return {
        "class_id": 0,
        "class_name": class_name,
        "confidence": conf,
        "x1": cx - half, "y1": cy - half, "x2": cx + half, "y2": cy + half,
        "track_id": None,
        "moving": None,
    }


def _pose(lat=10.0, lon=20.0, heading=0.0, tilt=45.0, alt=1.5, accuracy=3.0):
    return Pose(lat=lat, lon=lon, alt=alt, heading_deg=heading, tilt_deg=tilt, accuracy_m=accuracy)


def test_nearby_feed_gets_a_ghost_and_originator_does_not():
    pose_a = _pose(lat=10.00000, lon=20.00000, heading=0.0)
    pose_b = _pose(lat=10.00003, lon=20.00000, heading=180.0)  # ~3m north of A, facing back at it

    out_a = engine.process("feed_a", pose_a, [_det()])
    assert all(not d["is_ghost"] for d in out_a)

    out_b = engine.process("feed_b", pose_b, [])
    ghosts = [d for d in out_b if d["is_ghost"]]
    assert len(ghosts) == 1
    assert ghosts[0]["source_feed"] == "feed_a"
    assert ghosts[0]["class_name"] == "personnel"


def test_far_feed_gets_no_ghost():
    pose_a = _pose(lat=10.0, lon=20.0)
    # ~1.1km away in latitude -- far outside any reasonable fusion radius.
    pose_c = _pose(lat=10.01, lon=20.0)

    engine.process("feed_a", pose_a, [_det()])
    out_c = engine.process("feed_c", pose_c, [])
    assert not any(d["is_ghost"] for d in out_c)


def test_stale_detection_falls_outside_the_time_window():
    from fusion import config

    pose_a = _pose(lat=10.0, lon=20.0)
    pose_b = _pose(lat=10.00003, lon=20.0)

    engine.process("feed_a", pose_a, [_det()])
    time.sleep((config.FUSION_WINDOW_MS + 100) / 1000.0)
    out_b = engine.process("feed_b", pose_b, [])
    assert not any(d["is_ghost"] for d in out_b)


def test_feed_that_already_sees_the_object_gets_no_duplicate_ghost():
    pose_a = _pose(lat=10.0, lon=20.0)
    pose_b = _pose(lat=10.00003, lon=20.0)

    engine.process("feed_a", pose_a, [_det()])
    out_b = engine.process("feed_b", pose_b, [_det()])  # B saw it too
    assert not any(d["is_ghost"] for d in out_b)


def test_backproject_forward_project_roundtrip():
    pose = _pose(lat=10.0, lon=20.0, heading=90.0, tilt=45.0, alt=2.0)
    lat, lon = geo.backproject(pose, (0.5, 0.5))
    bearing, in_frame, pixel = geo.forward_project(pose, lat, lon)
    assert in_frame
    assert pixel is not None
    assert abs(pixel[0] - 0.5) < 1e-6


def test_no_pose_passes_detections_through_unchanged():
    dets = [_det()]
    out = engine.process("feed_a", None, dets)
    assert out is dets
