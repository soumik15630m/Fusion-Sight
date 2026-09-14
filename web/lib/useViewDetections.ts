"use client";

import { useEffect, useState } from "react";
import { wsBase } from "@/lib/config";
import type { DetectionResponse } from "@/lib/types";

/** Operator-side counterpart to a feed's own /ws/track: subscribes to
 * /ws/detections/{source_id} (detector/app/routers/pose.py) for the merged
 * detections (own + cross-feed ghosts) the feed's frames produced, so the
 * operator VideoTile can overlay boxes on the relayed video. Read-only -- no
 * frames captured or encoded here. */
export function useViewDetections(sourceId: string): DetectionResponse | null {
  const [detection, setDetection] = useState<DetectionResponse | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`${wsBase()}/ws/detections/${encodeURIComponent(sourceId)}`);
    ws.onmessage = (ev) => {
      try {
        setDetection(JSON.parse(ev.data));
      } catch {
        // ignore malformed payload
      }
    };
    return () => ws.close();
  }, [sourceId]);

  return detection;
}
