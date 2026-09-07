"use client";

import { useEffect, useRef, useState } from "react";
import { wsBase } from "@/lib/config";

interface Props {
  sourceId: string;
}

/** Subscribes to /ws/view/{source_id} (fusion/frame_relay.py) -- the same
 * already-decoded frames /ws/track or /webrtc/offer relayed, no second
 * capture/encode on the phone. */
export default function VideoTile({ sourceId }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [connected, setConnected] = useState(false);
  const lastUrlRef = useRef<string | null>(null);

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
    </div>
  );
}
