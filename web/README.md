# Web

Next.js client. Not yet implemented — will contain:

- `app/device/[sourceId]/page.tsx` — mobile-web capture + "AR" view for one
  feed: camera + GPS/orientation capture, sends video over `detector`'s
  existing `WS /ws/track/{sourceId}` and pose over `fusion`'s
  `WS /ws/pose/{sourceId}`, renders own detections plus ghost markers
  (with off-screen edge indicators) and a small map panel.
- `app/operator/page.tsx` — overview page: live video grid across all
  active feeds (via `fusion`'s frame relay) plus a shared map showing every
  feed's position and all correlated objects.
