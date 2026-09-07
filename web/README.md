# Web

Next.js (App Router) client. Two pages:

- `/device/[sourceId]` — the "AR" stand-in for one wearer/phone: captures this
  device's camera, GPS (`navigator.geolocation`) and orientation
  (`deviceorientation`/`deviceorientationabsolute`), streams video over
  `detector`'s `WS /ws/track/{sourceId}` and pose over `fusion`'s
  `WS /ws/pose/{sourceId}`, and renders the returned detections: solid boxes
  for this feed's own detections, dashed markers for in-frame ghosts, and an
  edge-of-frame arrow (from `bearing_deg`) for ghosts outside the current
  view. Includes a small map of nearby feeds (`GET /fusion/feeds`).
- `/operator` — grid of every active feed's live video (via `fusion`'s
  `WS /ws/view/{sourceId}` frame relay) plus the same map, showing every
  feed's position at once.

## Run

```bash
npm install
npm run dev
```

`next dev --hostname 0.0.0.0` (already in `package.json`'s `dev` script) is
required so phones on the same LAN can reach it — `localhost` only binds
loopback.

By default the client talks to the backend at
`http://<page's own hostname>:8000` (see `lib/config.ts`) — i.e. if you open
`http://192.168.1.20:3000/device/feed-a` on a phone, it assumes the backend
is at `192.168.1.20:8000`. Override with `NEXT_PUBLIC_BACKEND_HOST` /
`NEXT_PUBLIC_BACKEND_PORT` (e.g. in `.env.local`) if the backend runs
somewhere else.

## Demo checklist

1. Start the backend (`detector/README.md`) with `FUSION_ENABLED=1` (the
   default).
2. Start this app, find the host laptop's LAN IP.
3. On each phone: open `http://<lan-ip>:3000/device/<unique-id>`, tap
   **Start**, grant camera + location (+ motion/orientation on iOS)
   permissions.
4. Point two phones at the same general area from different angles — an
   object one phone's model misses but the other catches nearby should
   appear as a dashed ghost marker (or edge arrow if off-screen) on the
   phone that missed it.
5. Open `/operator` on a laptop/tablet to see every feed's live video and
   position on one screen.

## Known limitations (demo scope, not bugs)

- Orientation (compass heading, tilt) is read directly from the raw device
  sensor with no calibration step — accuracy varies a lot by device/browser.
- The map has no real tile caching; it needs internet access on the
  phone/laptop for OpenStreetMap tiles to load (the actual detection/fusion
  path does not need internet, only the visual map).
- No auth/access control anywhere in this stack — LAN-only demo assumption.
