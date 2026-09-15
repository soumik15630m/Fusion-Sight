"use client";

import { useEffect, useRef, useState } from "react";
import type { Detection } from "@/lib/types";

/** Per-frame YOLO output is jittery: boxes wobble a few pixels and objects
 * blink in and out as detection confidence crosses threshold, and ghosts
 * additionally flicker as GPS/proximity flips. This hook stabilises the render
 * without touching the detection/fusion logic:
 *   - identity across frames (track_id for own boxes; source_feed + world
 *     position for ghosts),
 *   - a short persistence window so a box that vanishes for a frame or two
 *     stays drawn instead of blinking,
 *   - EMA smoothing of box coordinates (and ghost bearing) so it glides
 *     instead of snapping.
 * It re-runs on every incoming frame, which is also when expiry is applied, so
 * stale boxes are dropped as long as frames keep arriving. */
const TTL_MS = 700; // keep a box alive this long after it was last seen
const EMA = 0.4; // position smoothing weight (0..1; lower = smoother/laggier)

interface Tracked {
  det: Detection;
  lastSeen: number;
}

function keyFor(d: Detection): string {
  if (d.track_id != null) return `o:${d.track_id}`;
  if (d.is_ghost) {
    // A feed can backfill several ghosts; separate them by originating feed and
    // rounded world position (~1m grid) rather than merging them all.
    const la = d.world_lat != null ? d.world_lat.toFixed(5) : "?";
    const lo = d.world_lon != null ? d.world_lon.toFixed(5) : "?";
    return `g:${d.source_feed ?? "?"}:${la}:${lo}`;
  }
  const cx = ((d.x1 + d.x2) / 2).toFixed(2);
  const cy = ((d.y1 + d.y2) / 2).toFixed(2);
  return `o:${d.class_id}:${cx}:${cy}`;
}

const lerp = (a: number, b: number, t: number) => a + t * (b - a);

/** Shortest-path angular interpolation, degrees. */
function lerpAngle(a: number, b: number, t: number): number {
  const diff = (((b - a) % 360) + 540) % 360 - 180;
  return (a + diff * t + 360) % 360;
}

export function useStableDetections(detections: Detection[]): Detection[] {
  const storeRef = useRef<Map<string, Tracked>>(new Map());
  const [out, setOut] = useState<Detection[]>([]);

  useEffect(() => {
    const now = Date.now();
    const store = storeRef.current;

    for (const d of detections) {
      const key = keyFor(d);
      const prev = store.get(key);
      if (prev) {
        const s = prev.det;
        const merged: Detection = {
          ...d,
          x1: lerp(s.x1, d.x1, EMA),
          y1: lerp(s.y1, d.y1, EMA),
          x2: lerp(s.x2, d.x2, EMA),
          y2: lerp(s.y2, d.y2, EMA),
          bearing_deg:
            d.bearing_deg != null && s.bearing_deg != null
              ? lerpAngle(s.bearing_deg, d.bearing_deg, EMA)
              : d.bearing_deg,
        };
        store.set(key, { det: merged, lastSeen: now });
      } else {
        store.set(key, { det: d, lastSeen: now });
      }
    }

    for (const [k, v] of store) {
      if (now - v.lastSeen > TTL_MS) store.delete(k);
    }

    setOut(Array.from(store.values(), (v) => v.det));
  }, [detections]);

  return out;
}
