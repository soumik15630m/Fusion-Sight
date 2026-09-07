"use client";

import { useEffect, useRef } from "react";
import type { Detection } from "@/lib/types";

interface Props {
  detections: Detection[];
  width: number;
  height: number;
  /** This device's current compass heading, needed to turn a ghost's
   * absolute bearing_deg into a screen-relative off-screen-indicator angle. */
  ownHeadingDeg: number;
}

const OWN_COLOR = "#00e5ff";
const GHOST_COLOR = "#ff2fd6";

/** Clamp a unit direction (angle measured from straight-up, clockwise) onto
 * the border of a width x height rectangle centered at its own center --
 * the standard "off-screen indicator" projection. */
function edgePoint(angleDeg: number, width: number, height: number): [number, number] {
  const rad = (angleDeg * Math.PI) / 180;
  const dx = Math.sin(rad);
  const dy = -Math.cos(rad);
  const cx = width / 2;
  const cy = height / 2;
  const scale = Math.min(
    dx !== 0 ? (cx - 12) / Math.abs(dx) : Infinity,
    dy !== 0 ? (cy - 12) / Math.abs(dy) : Infinity
  );
  return [cx + dx * scale, cy + dy * scale];
}

export default function DetectionOverlay({ detections, width, height, ownHeadingDeg }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    for (const det of detections) {
      const color = det.is_ghost ? GHOST_COLOR : OWN_COLOR;

      if (!det.is_ghost || det.in_frame !== false) {
        const x1 = det.x1 * width;
        const y1 = det.y1 * height;
        const x2 = det.x2 * width;
        const y2 = det.y2 * height;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        if (det.is_ghost) ctx.setLineDash([6, 4]);
        else ctx.setLineDash([]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = `${det.is_ghost ? "GHOST " : ""}${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "12px sans-serif";
        const textW = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - 16), textW + 6, 16);
        ctx.fillStyle = "#000";
        ctx.fillText(label, x1 + 3, Math.max(12, y1 - 4));
      } else if (det.bearing_deg != null) {
        // Off-screen ghost: draw an edge arrow pointing toward it, relative
        // to this device's own current heading.
        const relative = ((det.bearing_deg - ownHeadingDeg + 540) % 360) - 180;
        const [px, py] = edgePoint(relative, width, height);
        ctx.save();
        ctx.translate(px, py);
        ctx.rotate((relative * Math.PI) / 180);
        ctx.fillStyle = GHOST_COLOR;
        ctx.beginPath();
        ctx.moveTo(0, -14);
        ctx.lineTo(8, 6);
        ctx.lineTo(-8, 6);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
        ctx.font = "11px sans-serif";
        ctx.fillStyle = GHOST_COLOR;
        ctx.fillText(det.class_name, px - 20, py + 24);
      }
    }
  }, [detections, width, height, ownHeadingDeg]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    />
  );
}
