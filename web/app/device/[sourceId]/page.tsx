"use client";

import dynamic from "next/dynamic";
import { use, useState } from "react";
import DetectionOverlay from "@/components/DetectionOverlay";
import { useDeviceFeed } from "@/lib/useDeviceFeed";
import { useFusionFeeds } from "@/lib/useFusionFeeds";

// react-leaflet touches window/document at import time -- must be client-only,
// and dynamic() with ssr:false is how that's done in the App Router.
const MapPanel = dynamic(() => import("@/components/MapPanel"), { ssr: false });

const DISPLAY_WIDTH = 480;
const DISPLAY_HEIGHT = 360;

export default function DevicePage({ params }: { params: Promise<{ sourceId: string }> }) {
  const { sourceId } = use(params);
  const [view, setView] = useState<"ground" | "drone">("ground");
  const { videoRef, status, error, detection, pose, start, stop } = useDeviceFeed(sourceId, view);
  const feeds = useFusionFeeds();

  const detections = detection?.detections ?? [];
  const ghostCount = detections.filter((d) => d.is_ghost).length;

  return (
    <main style={{ padding: 16, maxWidth: 520, margin: "0 auto" }}>
      <h1 style={{ fontSize: 18 }}>
        {sourceId} <span style={{ color: "#888", fontWeight: 400 }}>({status})</span>
      </h1>

      <div
        style={{
          position: "relative",
          width: DISPLAY_WIDTH,
          height: DISPLAY_HEIGHT,
          background: "#000",
          margin: "0 auto",
        }}
      >
        <video
          ref={videoRef}
          muted
          playsInline
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
        <DetectionOverlay
          detections={detections}
          width={DISPLAY_WIDTH}
          height={DISPLAY_HEIGHT}
          ownHeadingDeg={pose?.heading_deg ?? 0}
        />
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 12, justifyContent: "center" }}>
        {status === "idle" || status === "error" ? (
          <button onClick={start} style={{ padding: "10px 20px", background: "#00e5ff", border: "none" }}>
            Start
          </button>
        ) : (
          <button onClick={stop} style={{ padding: "10px 20px", background: "#333", color: "#eee", border: "none" }}>
            Stop
          </button>
        )}
        <select
          value={view}
          onChange={(e) => setView(e.target.value as "ground" | "drone")}
          style={{ background: "#111", color: "#eee", border: "1px solid #333" }}
        >
          <option value="ground">ground</option>
          <option value="drone">drone</option>
        </select>
      </div>

      {error && <p style={{ color: "#ff5252", textAlign: "center" }}>{error}</p>}

      <p style={{ textAlign: "center", color: "#888", fontSize: 13 }}>
        {detections.length - ghostCount} own detection(s), {ghostCount} ghost(s) backfilled from other feeds
        {detection && ` · ${detection.inference_ms.toFixed(0)}ms`}
      </p>

      <h2 style={{ fontSize: 14, color: "#aaa", marginTop: 24 }}>Nearby feeds</h2>
      <MapPanel feeds={feeds} selfId={sourceId} />
    </main>
  );
}
