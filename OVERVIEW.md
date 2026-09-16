# FusionSight — Complete Project Overview

> This document is written for **both non-technical and technical readers**.
> The first half ("Part 1") explains what the project is and why it matters in
> plain language. The second half ("Part 2") is the technical reference —
> architecture, code, deployment, and operations. Skip to whichever you need.

---

# Part 1 — The plain-language version

## What is FusionSight?

FusionSight gives a **team of people shared eyesight**.

Imagine several people spread across an area, each wearing a smart camera (in
this demo, a phone stands in for AR glasses). Each camera automatically spots
things of interest — people and vehicles. Normally, each person only sees what
their *own* camera catches.

FusionSight changes that: **if one person's camera spots something that another
person's camera missed, it draws that object into the second person's view** —
even pointing an on-screen arrow toward it if it's just out of frame. So a
detail one teammate notices is instantly shared with everyone nearby who should
also be able to see it.

There's also a **command/operator screen** that shows every camera's live video
side-by-side, plus a **single map** with everyone's position and every object
that's been detected across the whole team.

## The problem it solves

A single camera (or a single person) misses things — because of angle,
distance, a moment of distraction, or something briefly out of view. When
several people are looking at roughly the same area, their blind spots don't
overlap. FusionSight **combines everyone's sightings into one shared picture**,
so the group effectively sees more than the sum of its individual cameras.

## How it works, in everyday terms

Think of it as three jobs:

1. **The eyes (each camera).** Every camera watches its scene and recognizes
   objects — "that's a person," "that's a vehicle." It also knows *where it is*
   (GPS) and *which way it's pointing* (compass + tilt).

2. **The brain (the fusion service).** All the cameras report what they see and
   where they are to one central "brain." Using each camera's location and
   aiming direction, the brain figures out the **real-world position** of each
   spotted object. If two cameras are looking at the same real-world spot and
   only one of them noticed an object there, the brain tells the *other* camera:
   "there's something here you missed" — a **ghost marker**.

3. **The display (each person's screen + the operator screen).** Each person's
   screen shows those ghost markers laid over their live camera view. The
   operator's screen shows all cameras and a shared map of every detected
   object.

## What you actually see

- **On a wearer's screen:** their normal camera view, with **magenta "ghost"
  boxes** for objects other cameras can see but theirs can't (with an edge arrow
  if the object is off to the side). It deliberately does **not** clutter their
  view with their own detections — only the *extra* information they'd otherwise
  be missing.

- **On the operator's screen:** a **grid of every camera's live video** (each
  with its own detections in cyan plus the shared ghosts in magenta), and a
  **map** showing every person by name and every detected object. Objects seen
  by **two or more** cameras (high confidence) are shown in one color; objects
  seen by only **one** camera (unconfirmed) in another.

## What a demo looks like

- A couple of **phones** act as the wearable cameras. Each person opens a web
  link, taps **Start**, and points their phone at the scene.
- A **laptop with a graphics card (GPU)** does the heavy "recognize objects"
  work.
- Another **laptop** publishes everything securely to the internet so the phones
  can connect from anywhere.
- Anyone can open the **operator link** on a laptop to watch the whole thing.

## Honest limitations (worth knowing up front)

- **Phone GPS is not precise, especially indoors.** Two people standing next to
  each other can appear several meters apart on the map. FusionSight is tuned to
  tolerate this (it treats nearby cameras as "in the same place"), but it cannot
  place people in a room with pinpoint accuracy — that's a physical limit of
  GPS, not a bug.
- **Matching is by location for now.** The current version decides "same object"
  by real-world position. Recognizing the *same specific object* by its
  appearance (visual re-identification) is a planned future upgrade.
- This is a **demo/prototype**, not certified hardware. Phones stand in for AR
  glasses; a laptop stands in for a real edge server.

## A quick glossary (for everyone)

| Term | Plain meaning |
|---|---|
| **Feed** | One camera's live stream (one person). |
| **Detection** | An object a camera recognized on its own. |
| **Ghost** | An object *another* camera saw, drawn into your view because you missed it. |
| **Fusion** | Combining all cameras' sightings into one shared picture. |
| **Pose** | Where a camera is and which way it's pointing (GPS + compass + tilt). |
| **Operator** | The overview screen showing all cameras + the shared map. |
| **Confirmed object** | Something 2+ cameras agree on (higher confidence). |

---

# Part 2 — The technical reference

## What it is, precisely

FusionSight is a **multi-feed object-detection system with cross-feed
geolocation fusion**. N camera feeds (phones as AR-glasses stand-ins) each run
independent object detection. A central fusion service back-projects each 2D
detection to a real-world ground coordinate using the feed's pose, correlates
detections across feeds by spatial proximity within a short time window, and
**backfills "ghost" detections** into feeds that missed an object another feed
saw nearby. A web client renders per-device AR overlays and an operator overview
(video wall + unified world map).

The detector recognizes four classes: `personnel`, `two_wheeler`,
`light_vehicle`, `heavy_vehicle`, in two model "views" (`ground` and `drone`).

## System architecture

The system is **three application services** behind one **edge** (reverse proxy
+ public tunnel):

```
                     phones + operator  (public HTTPS)
                              │
                              ▼
                    ┌───────────────────┐
                    │  EDGE             │  Caddy (one origin) + Cloudflare tunnel
                    │  routes by path   │  + Next.js web client
                    └───────────────────┘
             frames │ /ws/track,/ws/view        │ pose, ghosts, operator overlays, map
                    ▼                            ▼
          ┌───────────────────┐      ┌──────────────────────────────┐
          │ DETECTOR (GPU)    │─────▶│ FUSION (no GPU)              │
          │ YOLO inference    │ raw  │ pose_store + registry +      │
          │ + video relay     │ dets │ geolocation engine + relays  │
          └───────────────────┘      └──────────────────────────────┘
```

**Responsibilities**

- **detector** — inference only. Decodes each JPEG frame, runs YOLO, returns a
  compact status ack to the phone, republishes the frame to operator viewers
  (`/ws/view`), and streams the raw detections to fusion.
- **fusion** — all cross-feed state and geometry. Ingests poses (from phones)
  and detections (from the detector), back-projects detections to world
  coordinates, correlates across feeds, produces ghosts and a deduplicated world
  map, and fans results out to phones and operators.
- **web** — Next.js client: a per-device AR page (shows *only* ghosts) and an
  operator overview (video wall + map).
- **edge** — Caddy reverse proxy presenting a single origin, plus a Cloudflare
  tunnel for public HTTPS (phone camera + GPS require a secure origin).

## End-to-end data flow

```
phone ──frames──▶ EDGE ──▶ DETECTOR ──inference──▶ raw detections ──▶ FUSION
phone ──pose──────▶ EDGE ─────────────────────────────────────────▶ FUSION
FUSION ──ghosts──▶ EDGE ──▶ phone screen (AR overlay)
FUSION ──overlays─▶ EDGE ──▶ operator video tiles
FUSION ──world map─▶ EDGE ──▶ operator map
DETECTOR ──video──▶ EDGE ──▶ operator video tiles (frames)
```

Key design point — the **async ghost channel**: the phone displays *only*
ghosts, and those come from fusion on their own stream. The phone's own
detections are used only to feed fusion, so the detector→fusion hop is **off the
phone's critical latency path**.

## Core concepts (technical)

- **source_id** — stable identifier per feed (e.g. `alpha`). Keys the tracker
  state, pose history, and relays.
- **Pose** — `{lat, lon, alt, heading_deg, tilt_deg, accuracy_m, fov_deg, name}`.
  Sent on its own cadence (~2/s), independent of the video frame rate.
- **Back-projection** (`fusion/geo.py`) — turns a normalized pixel detection into
  a real-world lat/lon assuming a flat ground plane, using the feed's GPS +
  heading + tilt + horizontal FOV. Also does the inverse (forward projection)
  to place a ghost back into a feed's view and compute a bearing for the
  off-screen arrow.
- **Registry** (`fusion/registry.py`) — recent geolocated detections across all
  feeds, pruned to a sliding time window.
- **Ghost** — a detection another feed produced nearby that this feed's own
  model missed. Carries `is_ghost=true`, `source_feed`, `world_lat/lon`,
  `bearing_deg`, and `in_frame`.
- **World object** — clustered, deduplicated real-world object across feeds with
  a `confirmations` count (how many distinct feeds saw it), for the operator map.
- **Proximity match** — two things are "the same" if within a radius that scales
  with reported GPS accuracy, and their timestamps fall inside the fusion window.

## Technology stack

| Layer | Tech |
|---|---|
| Detection | Python, **Ultralytics YOLO**, **PyTorch (CUDA 12.6)**, OpenCV |
| Backend services | **FastAPI**, Uvicorn, WebSockets |
| Fusion | Pure-Python (stdlib) geometry — no ML deps, tiny image |
| Frontend | **Next.js 16** (React), **Leaflet** maps, WebSocket + Canvas overlays |
| Edge / networking | **Caddy** (reverse proxy), **Cloudflare Tunnel**, **Tailscale** (mesh) |
| Packaging | **Docker** / Docker Compose |
| GPU (tested) | NVIDIA RTX 3050 (any CUDA 12.x NVIDIA GPU works) |

## Repository layout

```
detector/                 GPU inference service (FastAPI)
  app/
    main.py               app wiring, /health, lifespan (loads model, starts fusion client)
    detector.py           YOLO load + per-feed track/motion state
    fusion_client.py      persistent outbound WS to the fusion service (+ preflight)
    frame_relay.py        fan-out of frames to operator viewers
    routers/
      track.py            /ws/track  — frames in, detections out, forward to fusion
      view.py             /ws/view   — operator video relay
      detect.py           /detect    — stateless HTTP detection
      webrtc.py           /webrtc    — optional low-latency transport
      exclude.py, train.py
  weights/                best.pt (ground) + drone_best.pt (drone)
  requirements.txt        pinned deps (Python 3.12, torch cu126)
  Dockerfile              CUDA 12.6 image
  multi_feed_fusion_spec.md   original design intent (Q&A spec)

fusion/                   Cross-feed correlation service (FastAPI, no GPU)
  main.py                 app + endpoints (pose, ingest, ghosts, detections, feeds, map)
  engine.py               correlate → ghosts + world_objects()
  registry.py             recent geolocated detections (sliding window)
  pose_store.py           per-feed pose history
  geo.py                  back/forward projection (spherical earth, flat ground)
  detection_relay.py      bounded fan-out (operator overlays + phone ghosts)
  config.py               tunables (radius, window, FOV, ...)
  schemas.py              PoseUpdate model
  Dockerfile              python:3.12-slim (tiny)

web/                      Next.js client
  app/
    device/[sourceId]/    per-device AR page (camera capture + ghost overlay)
    operator/             operator overview (video wall + map)
  components/             VideoTile, DetectionOverlay, MapPanel
  lib/                    hooks: useDeviceFeed, useGhostStream, useViewDetections,
                          useFusionFeeds, useWorldObjects, useStableDetections; config
  Dockerfile

deploy/
  backend/                2-laptop: detector + fusion together (GPU machine)
  detector/               3-laptop: detector alone
  fusion/                 3-laptop: fusion alone
  broadcast/              edge: web + Caddy + cloudflared (+ .env.example)
  Caddyfile               path routing to detector/fusion/web (env-driven upstreams)

docker-compose.yml        ALL-IN-ONE (everything on one machine)
DEPLOY.md                 deployment guide (all-in-one, 2-laptop, 3-laptop)
OVERVIEW.md               this document
README.md                 short intro
```

## HTTP / WebSocket endpoints

**Detector**

| Endpoint | Purpose |
|---|---|
| `WS /ws/track/{source_id}` | frames in (binary JPEG) → compact status ack; forwards detections to fusion |
| `WS /ws/view/{source_id}` | operator video: relayed JPEG frames |
| `POST /webrtc/offer/{source_id}` | optional WebRTC transport |
| `POST /detect`, `/detect/tracked` | stateless / stream HTTP detection |
| `GET /health` | model, device, `fusion_link {url, connected}` |

**Fusion**

| Endpoint | Purpose |
|---|---|
| `WS /ws/pose/{source_id}` | phone pose telemetry in |
| `WS /ws/ingest` | detector → raw detections in (tagged by source_id) |
| `WS /ws/ghosts/{source_id}` | phone AR: ghost markers out |
| `WS /ws/detections/{source_id}` | operator: merged own+ghost overlays out |
| `GET /fusion/feeds` | active feeds + pose + name (map) |
| `GET /fusion/detections` | deduplicated world objects + confirmations (map) |
| `GET /health` | fusion status + tuning values |

The **edge Caddy** presents all of these under one origin (routing `/ws/track`,
`/ws/view`, `/webrtc`, `/detect`, `/health` → detector; `/ws/pose`, `/ws/ghosts`,
`/ws/detections`, `/fusion/*` → fusion; everything else → web).

## Configuration (environment variables)

| Variable | Service | Default | Meaning |
|---|---|---|---|
| `BATTLESIGHT_DEVICE` | detector | `0` | GPU index, or `cpu` |
| `FUSION_WS_URL` | detector | `ws://fusion:8100/ws/ingest` | where to stream detections |
| `FUSION_ENABLED` | fusion | `1` | enable correlation |
| `FUSION_BASE_RADIUS_M` | fusion | `20` | base proximity radius (metres) |
| `FUSION_WINDOW_MS` | fusion | `1500` | "concurrent" time window |
| `FUSION_POSE_MAX_AGE_MS` | fusion | `5000` | how long a pose stays usable |
| `FUSION_DEFAULT_FOV_DEG` | fusion | `67` | assumed camera FOV if unknown |
| `NEXT_PUBLIC_BACKEND_SAME_ORIGIN` | web | `1` (in Docker) | talk to backend on the proxy origin |
| `DETECTOR_HOST` / `FUSION_HOST` | edge | — | backend addresses (mesh IPs) |

## Running it

All three modes are in **DEPLOY.md**; summary:

**All-in-one (one machine):**
```bash
docker compose up --build
```

**2-laptop (recommended for the demo):**
- Laptop A (GPU): `cd deploy/backend && docker compose up --build`
- Laptop B (edge): set `deploy/broadcast/.env` (`DETECTOR_HOST`/`FUSION_HOST` =
  Laptop A's address), then `cd deploy/broadcast && docker compose up --build`

**3-laptop:** `deploy/detector`, `deploy/fusion`, `deploy/broadcast` — one each.

The public URL is printed in the cloudflared logs; open `…/operator` on a
laptop and `…/device/<name>` on each phone.

## Networking & security

- **Between laptops:** a **Tailscale (WireGuard) mesh** — encrypted private IPs,
  works across networks, no port-forwarding. Services bind to the tailnet.
- **Public edge:** a **Cloudflare tunnel** provides HTTPS (mandatory for phone
  camera + GPS, which browsers only allow on a secure origin). For a stable URL
  and a login gate, use a **named tunnel + Cloudflare Access** (quick tunnels
  give a random URL that changes on restart).
- **Tunnel protocol:** cloudflared is pinned to **`--protocol http2`** (TCP)
  instead of QUIC/UDP, because UDP-throttling networks cause QUIC timeouts and
  churn the tunnel URL.

## Latency & stability engineering

- **Async ghost channel** keeps the cross-service hop off the phone's own-view
  path (see data flow).
- **Compact phone ack** — the detector returns only `{inference_ms, own_count,
  dims}` to the phone (which shows only ghosts), not the full detection list.
- **GPS smoothing** — the device low-pass-filters raw GPS so a stationary feed
  settles instead of jittering (which would flip proximity matches).
- **Detection smoothing** — the client keeps each box/ghost alive for a short
  persistence window and EMA-smooths its position, so overlays glide instead of
  flickering.
- **Fusion tuning** — a widened radius and longer window make same-room feeds
  correlate reliably despite GPS noise.
- **Biggest remaining lever:** every frame currently traverses the Cloudflare
  edge. For a same-LAN demo, serving the edge over **LAN HTTPS** (local trusted
  cert) removes that WAN hop (documented in DEPLOY.md).

## Health checks & troubleshooting

- **One-call chain check:** `curl <edge-url>/health` routes to the detector and
  reports `fusion_link.connected` — so it verifies edge→detector *and*
  detector→fusion at once. `connected:false` ⇒ the detector can't reach fusion.
- **Startup preflight:** the detector resolves `FUSION_WS_URL` on boot and prints
  a loud banner if it can't (the exact "wrong address ⇒ silent ghost loss" case).

| Symptom | Likely cause → fix |
|---|---|
| Operator video works, no ghosts | need 2+ phones near each other, both with GPS; check `fusion_link.connected` |
| `502 Bad gateway` at the tunnel | edge can't reach backend — check `.env` addresses + Windows firewall on the backend (allow 8000/8100) |
| Cloudflare **1033** / dead URL | tunnel churned — read the newest URL from `cloudflared` logs; ensure `--protocol http2` |
| Container can't reach a Tailscale MagicDNS name | Docker Desktop DNS limitation — use the tailnet **IP** in `.env` |
| People appear far apart in one room | inherent GPS error indoors — expected, mitigated by the widened radius |

## Design intent & history

`detector/multi_feed_fusion_spec.md` captures the original Q&A-derived spec:
pose transport as a parallel telemetry channel, proximity-scaled matching,
3D-anchored ghost markers with off-screen indicators, the per-feed-module code
layout, and what was scoped in/out for v1 (geolocation-only matching; visual
re-identification deferred as a fast-follow).

---

*FusionSight — shared sight for a distributed team. Built as an SIH prototype:
phones as AR glasses, a GPU laptop as the edge compute, correlation across feeds
turning each camera's blind spots into the group's shared awareness.*
