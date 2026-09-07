# Fusion

Cross-feed pose ingestion and geolocation correlation. Not yet implemented —
see the top-level `../multi_feed_fusion_spec.md`... (spec currently under
`detector/`, moving here as this module is built) and the approved
implementation plan for what this will contain:

- Pose telemetry ingestion (`WS /ws/pose/{source_id}`) — parallel channel to
  `detector`'s video WS, not piggybacked on frames.
- `PoseStore` — per-feed sliding-window pose history.
- `geo.py` — pixel <-> lat/lon back/forward projection.
- `DetectionRegistry` + `FusionEngine` — cross-feed geolocation matching
  (v1: geolocation-only, proximity scaled by reported GPS accuracy; visual
  re-ID deferred) and ghost-marker generation, called synchronously from
  `detector`'s per-frame path so it stays inside the existing latency
  budget.
- Live frame relay for the operator video grid.
