# Multi-Feed Cross-Correlation & AR Projection — Design Intent

Captured from a Q&A pass before implementation. This is intent/spec, not a
plan — no code has been written yet.

## Goal

BattleSight currently runs independent detection per feed (`source_id`),
each with its own tracker/motion state (`app/detector.py`), returning
normalised per-frame boxes (`app/schemas.py`). There is no cross-feed
correlation today.

Target: 4-5 feeds (scaling to N later), each worn by a different person on
AR glasses. If feed A's model detects an object that feed B/C/D's own model
misses, but geolocation says it's the same real-world object those feeds
should also be seeing (within proximity), inject a 3D world-anchored marker
into B/C/D's AR view at that object's real position — so a miss on one
feed's model is backfilled by the others.

## Decisions so far

1. **Camera pose per feed**: GPS + compass/IMU alongside each video stream.
   Used to back-project a 2D pixel detection to a real-world ground-plane
   coordinate (needs feed's own lat/lon/alt + heading/tilt + camera
   intrinsics/FOV).

2. **Cross-feed match criteria**: geolocation proximity **and** visual
   re-identification (embedding similarity), not class label alone. A match
   requires both the projected geo-positions to be within a distance
   threshold AND a visual appearance match above some similarity threshold.

3. **AR output for a "ghost" object** (seen by another feed, missed by this
   one): a **3D world-anchored marker**, projected into that wearer's live
   view using their own head pose/camera intrinsics — moves correctly as
   they turn their head, includes an off-screen/edge indicator when the
   object isn't currently in frame (not just a flat box drawn only when
   already in frame).

4. **Code layout**: one detector module per feed/model type sharing a common
   base (reusing the existing `Detector` class pattern, one instance per
   `source_id`/view already exists). New fusion logic lives in a separate
   module (e.g. `app/fusion/`) that consumes detections + geo-pose from all
   active feeds and does the cross-feed correlation and per-feed fan-out —
   it does not get folded into `app/detector.py` itself.

5. **Demo scope, not real AR hardware.** For this demo the capture side is a
   phone camera + phone GPS/IMU, and the display side is a web page
   (HTML/Next.js) standing in for AR glasses — both need to be built as part
   of this project:
   - **Phone capture app**: built from scratch. Streams video plus captures
     GPS/IMU.
   - **Pose transport**: a **parallel telemetry channel**, separate from the
     video stream, timestamped independently rather than attached to every
     frame message. Correlation between a video frame and its pose uses the
     sliding time-window match below.
   - **Web "AR" client**: one page per device/feed (split view — live video
     with overlaid boxes for its own detections plus distinctly-styled
     "ghost" markers for objects backfilled from other feeds, with an
     off-screen/edge indicator when the ghost object isn't in the current
     frame — and a shared top-down map panel alongside the video). Plus one
     operator overview page showing all feeds (or the map) at once, for
     demoing/debugging the fusion behavior in one place.

6. **Re-ID scope**: geolocation-only matching for v1. Visual re-ID
   (appearance embedding similarity) is deferred as a fast-follow once the
   geo-proximity pipeline is proven end-to-end — not required for the first
   working version, despite decision 2 above being the eventual target.

7. **Proximity threshold**: scaled by each feed's estimated pose uncertainty
   (not a single fixed number) — a feed reporting worse GPS accuracy gets a
   looser match radius. Falls back to a configurable default distance if a
   feed doesn't report accuracy/HDOP.

8. **Feed time-sync**: sliding time-window match. Detections (and their
   associated pose) from different feeds are treated as "concurrent" if
   their timestamps fall within a short window (on the order of a few
   hundred ms) of each other; correlation matches against the most recent
   detection per feed inside that window.

9. **Fusion latency budget**: must fit inside the existing per-frame
   glass-to-overlay budget (~100-190 ms end to end, see `README.md` §5) —
   not a separate relaxed async pass. Ghost-marker correlation and fan-out
   needs to happen fast enough to land in the same frame's response as each
   feed's own detections, which constrains how heavy re-ID (decision 6, once
   added) or any per-frame cross-feed lookup can be.

## Still open (revisit before implementation)

- Exact wire format for the phone app's pose telemetry channel and how the
  backend correlates it to a specific video frame within the sliding window.
- Camera intrinsics/FOV capture from the phone (needed for accurate
  pixel→ground back-projection), and how the operator/map page gets each
  feed's live position to plot.
- What "pose uncertainty" input is actually available from a phone (GPS
  accuracy/HDOP field, or something coarser) to scale the proximity
  threshold by.
- Given fusion must fit the existing latency budget, whether N=4-5 feeds'
  worth of pairwise correlation per frame is cheap enough as a naive
  loop, or needs a spatial index once feed count scales up.
