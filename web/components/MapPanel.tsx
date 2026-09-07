"use client";

import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import type { FusionFeeds } from "@/lib/types";
import "leaflet/dist/leaflet.css";

// CircleMarker instead of the default Marker: the default Leaflet marker
// icon is a set of PNG assets that need webpack/asset-path wiring to load
// correctly under Next.js, and a plain circle is enough to show "a feed is
// here" for this demo.

interface Props {
  feeds: FusionFeeds;
  selfId?: string;
  height?: number;
}

export default function MapPanel({ feeds, selfId, height = 240 }: Props) {
  const entries = Object.entries(feeds);
  const center: [number, number] = entries.length
    ? [entries[0][1].lat, entries[0][1].lon]
    : [0, 0];

  if (entries.length === 0) {
    return (
      <div
        style={{
          height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#111",
          color: "#888",
          fontSize: 13,
        }}
      >
        no feeds with a live pose yet
      </div>
    );
  }

  return (
    <MapContainer center={center} zoom={19} style={{ height, width: "100%" }}>
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {entries.map(([sourceId, feed]) => (
        <CircleMarker
          key={sourceId}
          center={[feed.lat, feed.lon]}
          radius={8}
          pathOptions={{
            color: sourceId === selfId ? "#00e5ff" : "#ff5252",
            fillColor: sourceId === selfId ? "#00e5ff" : "#ff5252",
            fillOpacity: 0.8,
          }}
        >
          <Popup>
            <strong>{sourceId}</strong>
            <br />
            heading {feed.heading_deg.toFixed(0)}°
            {feed.accuracy_m != null && <> · ±{feed.accuracy_m.toFixed(1)}m</>}
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
