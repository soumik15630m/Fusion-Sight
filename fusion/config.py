"""Fusion config -- same flat os.getenv() pattern as detector/app/config.py."""
import os

FUSION_ENABLED = os.getenv("FUSION_ENABLED", "1") not in ("0", "false", "False")

# How long a pose/detection stays eligible for cross-feed matching. Feeds
# aren't frame-aligned, so "concurrent" means "within this many ms of each
# other", not "same frame".
FUSION_WINDOW_MS = int(os.getenv("FUSION_WINDOW_MS", "400"))

# Proximity radius (meters) two detections must fall within to count as the
# same real-world object. v1 is geolocation-only (no visual re-ID), so this
# radius is the entire matching decision -- see FUSION_UNCERTAINTY_MULTIPLIER
# for why it isn't a flat constant.
FUSION_BASE_RADIUS_M = float(os.getenv("FUSION_BASE_RADIUS_M", "8.0"))

# Consumer phone GPS reports ~3-5m error, and that error compounds between two
# independent feeds. Radius scales with each feed's own reported
# accuracy_m (phones expose this via the Geolocation API) so a feed with a bad
# fix gets a wider tolerance instead of a fixed one that's wrong for everyone.
FUSION_UNCERTAINTY_MULTIPLIER = float(os.getenv("FUSION_UNCERTAINTY_MULTIPLIER", "1.5"))

# Fallback horizontal field of view (degrees) for back/forward-projection when
# a feed doesn't report its own camera FOV. Typical phone main camera ~65-70.
FUSION_DEFAULT_FOV_DEG = float(os.getenv("FUSION_DEFAULT_FOV_DEG", "67.0"))

# If a detection's line of sight doesn't intersect the ground plane (camera
# pointed at/above the horizon -- tilt <= 0 after accounting for pose), there
# is no real ground-plane fix to compute. Falling back to a fixed assumed
# distance keeps the pipeline from just dropping the detection, at the cost of
# an approximate position -- acceptable for a demo, flagged here as the known
# limitation rather than silently wrong.
FUSION_FALLBACK_DISTANCE_M = float(os.getenv("FUSION_FALLBACK_DISTANCE_M", "25.0"))

# Pose history kept per feed (for sliding-window lookup) and how stale a pose
# can be before it's dropped from matching entirely.
FUSION_POSE_HISTORY = int(os.getenv("FUSION_POSE_HISTORY", "50"))
FUSION_POSE_MAX_AGE_MS = int(os.getenv("FUSION_POSE_MAX_AGE_MS", "2000"))
