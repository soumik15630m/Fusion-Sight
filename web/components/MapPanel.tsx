"use client";

import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import type { FusionFeeds, WorldObject } from "@/lib/types";
import "leaflet/dist/leaflet.css";

// CircleMarker instead of the default Marker: the default Leaflet marker
// icon is a set of PNG assets that need webpack/asset-path wiring to load
// correctly under Next.js, and a plain circle is enough to show "a feed is
// here" for this demo.

interface Props {
  feeds: FusionFeeds;
  selfId?: string;
  height?: number;
  /** Operator unified map only: deduplicated real-world objects across all
   * feeds. Color-coded by confirmations -- cross-confirmed (2+ feeds) vs a
   * single-feed sighting. */
  objects?: WorldObject[];
}

// Confirmed = seen by 2+ feeds (the fused/merged picture); unconfirmed = a
// lone sighting one feed reported and no other feed corroborated.
const CONFIRMED_COLOR = "#00e676";
const UNCONFIRMED_COLOR = "#ffb300";

export default function MapPanel({ feeds, selfId, height = 240, objects = [] }: Props) {
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
            <strong>{feed.name || sourceId}</strong>
            {feed.name && feed.name !== sourceId && (
              <>
                <br />
                <span style={{ color: "#888" }}>{sourceId}</span>
              </>
            )}
            <br />
            heading {feed.heading_deg.toFixed(0)}°
            {feed.accuracy_m != null && <> · ±{feed.accuracy_m.toFixed(1)}m</>}
          </Popup>
        </CircleMarker>
      ))}
      {objects.map((obj, i) => {
        const confirmed = obj.confirmations >= 2;
        const color = confirmed ? CONFIRMED_COLOR : UNCONFIRMED_COLOR;
        return (
          <CircleMarker
            key={`obj-${i}`}
            center={[obj.lat, obj.lon]}
            radius={6}
            pathOptions={{ color, fillColor: color, fillOpacity: 0.6, weight: confirmed ? 3 : 1 }}
          >
            <Popup>
              <strong>{obj.class_name}</strong>
              <br />
              {confirmed
                ? `confirmed by ${obj.confirmations} feeds`
                : "single-feed sighting (unconfirmed)"}
              <br />
              {obj.sources.join(", ")}
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
