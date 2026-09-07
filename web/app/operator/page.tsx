"use client";

import dynamic from "next/dynamic";
import VideoTile from "@/components/VideoTile";
import { useFusionFeeds } from "@/lib/useFusionFeeds";

const MapPanel = dynamic(() => import("@/components/MapPanel"), { ssr: false });

export default function OperatorPage() {
  const feeds = useFusionFeeds();
  const sourceIds = Object.keys(feeds);

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
          <VideoTile key={id} sourceId={id} />
        ))}
        {sourceIds.length === 0 && <p style={{ color: "#666" }}>Waiting for feeds...</p>}
      </div>

      <h2 style={{ fontSize: 14, color: "#aaa", marginTop: 24 }}>All feed positions</h2>
      <MapPanel feeds={feeds} height={360} />
    </main>
  );
}
