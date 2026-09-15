"use client";

import { useEffect, useState } from "react";
import { wsBase } from "@/lib/config";
import type { Detection } from "@/lib/types";

/** The wearer's AR ghosts now arrive on their own channel from the fusion
 * service (/ws/ghosts/{source_id}), not inside the /ws/track response -- the
 * detector no longer runs fusion. Same-origin, so it's routed to the fusion
 * laptop by the edge proxy. */
export function useGhostStream(sourceId: string): Detection[] {
  const [ghosts, setGhosts] = useState<Detection[]>([]);

  useEffect(() => {
    const ws = new WebSocket(`${wsBase()}/ws/ghosts/${encodeURIComponent(sourceId)}`);
    ws.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        setGhosts(payload.detections ?? []);
      } catch {
        // ignore malformed payload
      }
    };
    return () => ws.close();
  }, [sourceId]);

  return ghosts;
}
