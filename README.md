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

## Run it end to end

```bash
# 1. backend (detector + fusion, one FastAPI process)
cd detector && uvicorn app.main:app --host 0.0.0.0 --port 8000
# 2. client
cd web && npm install && npm run dev
```

Then open `http://<this machine's LAN IP>:3000/device/<id>` on each phone
and `http://<lan-ip>:3000/operator` on a laptop/tablet. See `web/README.md`
for the full demo checklist and `Dockerfile` for a containerized backend.

## Status

- `detector/` — built and working standalone (single-feed detection,
  tracking, training).
- `fusion/` — pose ingestion, geolocation correlation, ghost-marker
  generation, and live-video relay, wired into `detector`'s per-frame path.
  Verified via `fusion/tests/test_fusion.py` and end-to-end in a Docker
  container (see `Dockerfile`).
- `web/` — Next.js client: per-device capture/AR page and operator overview.
