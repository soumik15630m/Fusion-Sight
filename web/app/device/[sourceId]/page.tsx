"use client";

import dynamic from "next/dynamic";
import { use, useEffect, useState } from "react";
import DetectionOverlay from "@/components/DetectionOverlay";
import { useDeviceFeed } from "@/lib/useDeviceFeed";
import { useFusionFeeds } from "@/lib/useFusionFeeds";
import { useGhostStream } from "@/lib/useGhostStream";
import { useStableDetections } from "@/lib/useStableDetections";

// react-leaflet touches window/document at import time -- must be client-only,
// and dynamic() with ssr:false is how that's done in the App Router.
const MapPanel = dynamic(() => import("@/components/MapPanel"), { ssr: false });

const DISPLAY_WIDTH = 480;
const DISPLAY_HEIGHT = 360;

export default function DevicePage({ params }: { params: Promise<{ sourceId: string }> }) {
  const { sourceId } = use(params);
  const [view, setView] = useState<"ground" | "drone">("ground");

  // The wearer's display name, shown on the operator page instead of the raw
  // source_id. Defaults to the source_id, persisted per device so a reload keeps
  // it, and sent with every pose update.
  const [name, setName] = useState(sourceId);
  useEffect(() => {
    const saved = window.localStorage.getItem(`feed-name:${sourceId}`);
    if (saved) setName(saved);
  }, [sourceId]);
  useEffect(() => {
    window.localStorage.setItem(`feed-name:${sourceId}`, name);
  }, [sourceId, name]);

  const { videoRef, status, error, detection, pose, start, stop } = useDeviceFeed(
    sourceId,
    view,
    name
  );
  const feeds = useFusionFeeds();

  // This screen stands in for the wearer's AR glasses: it shows ONLY the ghost
  // markers backfilled from other feeds (the augmentation). Those now arrive on
  // their own channel from the fusion service (useGhostStream); the /ws/track
  // response carries only this feed's own detections (status/debug), which exist
  // just to feed other feeds' fusion. The operator page reviews own boxes.
  const ownCount = detection?.own_count ?? 0;
  const rawGhosts = useGhostStream(sourceId);
  const ghosts = useStableDetections(rawGhosts);

  return (
    <main style={{ padding: 16, maxWidth: 520, margin: "0 auto" }}>
      <h1 style={{ fontSize: 18 }}>
        {name || sourceId}{" "}
        <span style={{ color: "#888", fontWeight: 400, fontSize: 13 }}>
          {name && name !== sourceId ? `${sourceId} · ` : ""}
          {status}
        </span>
      </h1>

      <label style={{ display: "block", color: "#888", fontSize: 13, marginBottom: 8 }}>
        Display name{" "}
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={sourceId}
          style={{
            background: "#111",
            color: "#eee",
            border: "1px solid #333",
            padding: "6px 8px",
            marginLeft: 4,
          }}
        />
      </label>

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
          detections={ghosts}
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
        {ghosts.length} ghost(s) backfilled from other feeds
        <span style={{ color: "#555" }}> · {ownCount} own detection(s) sent to fusion (not shown here)</span>
        {detection && ` · ${detection.inference_ms.toFixed(0)}ms`}
      </p>

      <h2 style={{ fontSize: 14, color: "#aaa", marginTop: 24 }}>Nearby feeds</h2>
      <MapPanel feeds={feeds} selfId={sourceId} />
    </main>
  );
}
