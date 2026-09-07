"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const [sourceId, setSourceId] = useState("feed-a");
  const router = useRouter();

  return (
    <main style={{ padding: 24, maxWidth: 480, margin: "0 auto" }}>
      <h1>FusionSight</h1>
      <p style={{ color: "#aaa" }}>
        Multi-feed cross-correlated detection demo. Open a device page on each phone
        (different source id per phone), and the operator page on a laptop/tablet.
      </p>

      <h2>Device (phone)</h2>
      <p style={{ color: "#888", fontSize: 13 }}>
        Captures this phone&apos;s camera + GPS/orientation, streams to the backend,
        and shows this feed&apos;s own detections plus any ghost markers backfilled
        from nearby feeds.
      </p>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={sourceId}
          onChange={(e) => setSourceId(e.target.value)}
          placeholder="source id, e.g. feed-a"
          style={{ flex: 1, padding: 8, background: "#111", color: "#eee", border: "1px solid #333" }}
        />
        <button
          onClick={() => router.push(`/device/${encodeURIComponent(sourceId)}`)}
          style={{ padding: "8px 16px", background: "#00e5ff", border: "none", cursor: "pointer" }}
        >
          Open
        </button>
      </div>

      <h2 style={{ marginTop: 32 }}>Operator overview</h2>
      <p style={{ color: "#888", fontSize: 13 }}>
        Live video grid of every active feed, plus a shared map of all feed
        positions and correlated objects.
      </p>
      <a href="/operator">
        <button style={{ padding: "8px 16px", background: "#333", color: "#eee", border: "none", cursor: "pointer" }}>
          Open operator view
        </button>
      </a>
    </main>
  );
}
