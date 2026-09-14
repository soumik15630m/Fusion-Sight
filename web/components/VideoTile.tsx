"use client";

import { useEffect, useRef, useState } from "react";
import DetectionOverlay from "@/components/DetectionOverlay";
import { wsBase } from "@/lib/config";
import { useViewDetections } from "@/lib/useViewDetections";

interface Props {
  sourceId: string;
  /** This feed's current compass heading, so off-screen ghost arrows point
   * correctly. From the operator page's /fusion/feeds poll. */
  headingDeg?: number;
}

// Overlay drawing resolution -- boxes are normalised so this only sets the
// canvas's internal pixel grid; CSS stretches it to the tile.
const OVERLAY_WIDTH = 320;
const OVERLAY_HEIGHT = 240;

/** Subscribes to /ws/view/{source_id} (fusion/frame_relay.py) for the
 * already-decoded frames /ws/track or /webrtc/offer relayed, and to
 * /ws/detections/{source_id} for the merged detections those frames produced
 * -- so the operator sees every feed's own boxes AND the cross-feed ghosts,
 * the fusion the capture device never draws for itself. No second
 * capture/encode on the phone. */
export default function VideoTile({ sourceId, headingDeg = 0 }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [connected, setConnected] = useState(false);
  const lastUrlRef = useRef<string | null>(null);
  const detection = useViewDetections(sourceId);
  const detections = detection?.detections ?? [];
  const ghostCount = detections.filter((d) => d.is_ghost).length;

  useEffect(() => {
    const ws = new WebSocket(`${wsBase()}/ws/view/${encodeURIComponent(sourceId)}`);
    ws.binaryType = "blob";
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (ev) => {
      const url = URL.createObjectURL(ev.data as Blob);
      if (imgRef.current) imgRef.current.src = url;
      if (lastUrlRef.current) URL.revokeObjectURL(lastUrlRef.current);
      lastUrlRef.current = url;
    };
    return () => {
      ws.close();
      if (lastUrlRef.current) URL.revokeObjectURL(lastUrlRef.current);
    };
  }, [sourceId]);

  return (
    <div style={{ position: "relative", background: "#000", aspectRatio: "4 / 3" }}>
      <img ref={imgRef} style={{ width: "100%", height: "100%", objectFit: "cover" }} alt={sourceId} />
      <DetectionOverlay
        detections={detections}
        width={OVERLAY_WIDTH}
        height={OVERLAY_HEIGHT}
        ownHeadingDeg={headingDeg}
      />
      <div
        style={{
          position: "absolute",
          top: 4,
          left: 4,
          background: "rgba(0,0,0,0.6)",
          color: connected ? "#00e5ff" : "#888",
          fontSize: 12,
          padding: "2px 6px",
        }}
      >
        {sourceId} {connected ? "" : "(no signal)"}
      </div>
      {ghostCount > 0 && (
        <div
          style={{
            position: "absolute",
            top: 4,
            right: 4,
            background: "rgba(0,0,0,0.6)",
            color: "#ff2fd6",
            fontSize: 12,
            padding: "2px 6px",
          }}
        >
          {ghostCount} ghost{ghostCount === 1 ? "" : "s"}
        </div>
      )}
    </div>
  );
}
