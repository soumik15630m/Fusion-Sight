# FusionSight

Multi-feed object detection with cross-feed geolocation fusion. 4-5 camera
feeds (phones standing in for AR glasses) each run independent object
detection; feeds that miss an object another feed detected nearby get a
"ghost" marker for it via geolocation correlation.

## Layout

```
detector/   YOLO26 detection service (FastAPI) — per-feed inference, tracking,
            motion filtering. See detector/README.md.
fusion/     Cross-feed pose ingestion + geolocation correlation engine.
            See fusion/README.md.
web/        Next.js client — per-device "AR" page + operator overview with a
            live video grid and shared map. See web/README.md.
```

## Design intent

`multi_feed_fusion_spec.md` (under `detector/` for now) captures the
Q&A-derived spec for the fusion feature: pose transport, matching criteria,
AR marker behavior, and what's in/out of scope for v1.

## Status

- `detector/` — built and working standalone (single-feed detection,
  tracking, training). See `detector/README.md` for setup and run
  instructions.
- `fusion/`, `web/` — not yet implemented.
