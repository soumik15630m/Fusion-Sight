"use client";

import { useEffect, useState } from "react";
import { httpBase } from "@/lib/config";
import type { WorldObject } from "@/lib/types";

/** Polls GET /fusion/detections for the operator's unified world map -- the
 * deduplicated objects detected across all feeds. Low-rate like
 * useFusionFeeds (the map doesn't need per-frame updates). */
export function useWorldObjects(intervalMs = 1000): WorldObject[] {
  const [objects, setObjects] = useState<WorldObject[]>([]);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const res = await fetch(`${httpBase()}/fusion/detections`);
        if (!cancelled && res.ok) {
          const data = await res.json();
          setObjects(data.objects ?? []);
        }
      } catch {
        // backend unreachable/restarting -- keep last known objects, retry next tick
      }
    }
    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return objects;
}
