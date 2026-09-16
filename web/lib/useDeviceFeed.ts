"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { wsBase } from "@/lib/config";
import type { TrackAck } from "@/lib/types";

const FRAME_INTERVAL_MS = 200; // ~5 fps upload -- see detector/README.md's latency budget notes
const FRAME_MAX_WIDTH = 640; // downscale before encode: uplink size is the dominant latency lever (README "Live feed latency")
const POSE_INTERVAL_MS = 500;

// Raw phone GPS jumps a few metres frame-to-frame even when the phone is still,
// which makes the map dot jitter and flips cross-feed proximity on/off (ghosts
// flicker). A low-pass filter on lat/lon settles a stationary feed to a steady
// point; the lag it adds while walking is minor at this smoothing strength.
const GPS_SMOOTHING = 0.25; // EMA weight for each new fix (0..1; lower = smoother/laggier)

export type CaptureStatus = "idle" | "starting" | "live" | "error";

interface PoseState {
  lat: number;
  lon: number;
  accuracy_m?: number;
  heading_deg: number;
  tilt_deg: number;
}

/** Orientation is inherently approximate here -- there is no calibration
 * step, just the raw sensor reading translated into this project's
 * heading/tilt convention (0=north clockwise; tilt positive = looking
 * down). Good enough for a demo, flagged rather than hidden -- the backend
 * (fusion/geo.py) already documents its own simplifying assumptions in the
 * same spirit (flat ground, spherical earth, an assumed FOV). */
function readHeadingDeg(e: DeviceOrientationEvent): number | null {
  const webkit = (e as unknown as { webkitCompassHeading?: number }).webkitCompassHeading;
  if (typeof webkit === "number") return webkit;
  if (e.absolute && e.alpha != null) return (360 - e.alpha) % 360;
  return null;
}

function readTiltDeg(e: DeviceOrientationEvent): number | null {
  if (e.beta == null) return null;
  return 90 - e.beta;
}

export function useDeviceFeed(
  sourceId: string,
  view: "ground" | "drone" = "ground",
  name = ""
) {
  const [status, setStatus] = useState<CaptureStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [detection, setDetection] = useState<TrackAck | null>(null);
  const [pose, setPose] = useState<PoseState | null>(null);

  // Latest name in a ref so the running pose pump sends edits without needing to
  // restart the capture (start() isn't rebuilt when the label changes).
  const nameRef = useRef(name);
  useEffect(() => {
    nameRef.current = name;
  }, [name]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const trackWsRef = useRef<WebSocket | null>(null);
  const poseWsRef = useRef<WebSocket | null>(null);
  const framePumpRef = useRef<number | null>(null);
  const posePumpRef = useRef<number | null>(null);
  const geoWatchRef = useRef<number | null>(null);
  const poseRef = useRef<PoseState | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const gpsSmoothRef = useRef<{ lat: number; lon: number } | null>(null);

  const start = useCallback(async () => {
    setStatus("starting");
    setError(null);

    try {
      // iOS Safari gates both camera and orientation behind a user gesture;
      // this is called from a click handler for that reason.
      const OrientationEventCtor = window.DeviceOrientationEvent as unknown as {
        requestPermission?: () => Promise<"granted" | "denied">;
      };
      if (typeof OrientationEventCtor?.requestPermission === "function") {
        await OrientationEventCtor.requestPermission().catch(() => undefined);
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const canvas = document.createElement("canvas");
      canvasRef.current = canvas;

      const trackWs = new WebSocket(`${wsBase()}/ws/track/${encodeURIComponent(sourceId)}?view=${view}`);
      trackWs.onmessage = (ev) => {
        try {
          setDetection(JSON.parse(ev.data));
        } catch {
          // ignore malformed frame
        }
      };
      trackWs.onerror = () => setError("track socket error");
      trackWsRef.current = trackWs;

      const poseWs = new WebSocket(`${wsBase()}/ws/pose/${encodeURIComponent(sourceId)}`);
      poseWs.onerror = () => setError("pose socket error");
      poseWsRef.current = poseWs;

      if ("geolocation" in navigator) {
        geoWatchRef.current = navigator.geolocation.watchPosition(
          (pos) => {
            const prev = poseRef.current;
            // Low-pass the raw fix before it becomes this feed's position.
            const sm = gpsSmoothRef.current;
            const lat = sm
              ? sm.lat + GPS_SMOOTHING * (pos.coords.latitude - sm.lat)
              : pos.coords.latitude;
            const lon = sm
              ? sm.lon + GPS_SMOOTHING * (pos.coords.longitude - sm.lon)
              : pos.coords.longitude;
            gpsSmoothRef.current = { lat, lon };
            const next: PoseState = {
              lat,
              lon,
              accuracy_m: pos.coords.accuracy,
              heading_deg: prev?.heading_deg ?? 0,
              tilt_deg: prev?.tilt_deg ?? 0,
            };
            poseRef.current = next;
            setPose(next);
          },
          () => setError("geolocation unavailable/denied"),
          { enableHighAccuracy: true, maximumAge: 1000 }
        );
      }

      const onOrientation = (e: DeviceOrientationEvent) => {
        const heading = readHeadingDeg(e);
        const tilt = readTiltDeg(e);
        if (heading == null && tilt == null) return;
        const prev = poseRef.current;
        if (!prev) return; // no GPS fix yet -- nothing to attach orientation to
        const next: PoseState = {
          ...prev,
          heading_deg: heading ?? prev.heading_deg,
          tilt_deg: tilt ?? prev.tilt_deg,
        };
        poseRef.current = next;
        setPose(next);
      };
      window.addEventListener("deviceorientationabsolute", onOrientation as EventListener, true);
      window.addEventListener("deviceorientation", onOrientation as EventListener, true);

      framePumpRef.current = window.setInterval(() => {
        const video = videoRef.current;
        const cv = canvasRef.current;
        const ws = trackWsRef.current;
        if (!video || !cv || !ws || ws.readyState !== WebSocket.OPEN) return;
        if (video.videoWidth === 0) return;

        const scale = Math.min(1, FRAME_MAX_WIDTH / video.videoWidth);
        cv.width = Math.round(video.videoWidth * scale);
        cv.height = Math.round(video.videoHeight * scale);
        const ctx = cv.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, cv.width, cv.height);
        cv.toBlob(
          (blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) blob.arrayBuffer().then((buf) => ws.send(buf));
          },
          "image/jpeg",
          0.7
        );
      }, FRAME_INTERVAL_MS);

      posePumpRef.current = window.setInterval(() => {
        const ws = poseWsRef.current;
        const p = poseRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN || !p) return;
        ws.send(
          JSON.stringify({
            lat: p.lat,
            lon: p.lon,
            heading_deg: p.heading_deg,
            tilt_deg: p.tilt_deg,
            accuracy_m: p.accuracy_m,
            name: nameRef.current || undefined,
          })
        );
      }, POSE_INTERVAL_MS);

      setStatus("live");

      return () => {
        window.removeEventListener("deviceorientationabsolute", onOrientation as EventListener, true);
        window.removeEventListener("deviceorientation", onOrientation as EventListener, true);
      };
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [sourceId, view]);

  const stop = useCallback(() => {
    if (framePumpRef.current) clearInterval(framePumpRef.current);
    if (posePumpRef.current) clearInterval(posePumpRef.current);
    if (geoWatchRef.current != null) navigator.geolocation.clearWatch(geoWatchRef.current);
    trackWsRef.current?.close();
    poseWsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    gpsSmoothRef.current = null;
    setStatus("idle");
  }, []);

  return { videoRef, status, error, detection, pose, start, stop };
}
