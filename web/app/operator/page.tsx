"use client";

import dynamic from "next/dynamic";
import VideoTile from "@/components/VideoTile";
import { useFusionFeeds } from "@/lib/useFusionFeeds";
import { useWorldObjects } from "@/lib/useWorldObjects";

const MapPanel = dynamic(() => import("@/components/MapPanel"), { ssr: false });

export default function OperatorPage() {
  const feeds = useFusionFeeds();
  const worldObjects = useWorldObjects();
  const sourceIds = Object.keys(feeds);
  const confirmedCount = worldObjects.filter((o) => o.confirmations >= 2).length;

  return (
    <main style={{ padding: 16 }}>
      <h1 style={{ fontSize: 18 }}>Operator overview</h1>
      <p style={{ color: "#888", fontSize: 13 }}>
        {sourceIds.length} active feed(s). A feed appears here once its device page has sent at
        least one pose update.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
          gap: 8,
          marginTop: 12,
        }}
      >
        {sourceIds.map((id) => (
          <VideoTile
            key={id}
            sourceId={id}
            name={feeds[id]?.name}
            headingDeg={feeds[id]?.heading_deg ?? 0}
          />
        ))}
        {sourceIds.length === 0 && <p style={{ color: "#666" }}>Waiting for feeds...</p>}
      </div>

      <h2 style={{ fontSize: 14, color: "#aaa", marginTop: 24 }}>Unified world map</h2>
      <p style={{ color: "#888", fontSize: 13, marginTop: 0 }}>
        {worldObjects.length} object(s) detected across all feeds · {confirmedCount} cross-confirmed.{" "}
        <span style={{ color: "#00e676" }}>● confirmed (2+ feeds)</span>{" "}
        <span style={{ color: "#ffb300" }}>● single-feed</span>{" "}
        <span style={{ color: "#ff5252" }}>● camera</span>
      </p>
      <MapPanel feeds={feeds} objects={worldObjects} height={360} />
    </main>
  );
}
