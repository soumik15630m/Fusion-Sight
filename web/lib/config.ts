// Backend host resolution. Phones on the demo LAN load this page from
// whatever host serves the Next.js app, but the FastAPI backend is a
// separate process/port -- default to "same hostname the page was loaded
// from, port 8000" (works when both run on one laptop and phones just hit
// that laptop's LAN IP for both), overridable via NEXT_PUBLIC_BACKEND_HOST
// for a split deployment (e.g. backend on a different machine).
const BACKEND_PORT = process.env.NEXT_PUBLIC_BACKEND_PORT || "8000";

// Same-origin mode: the backend is reached at the exact host:port the page was
// served from (no separate :8000). This is how the Docker/Cloudflare-tunnel
// deployment works -- a reverse proxy (deploy/Caddyfile) serves the web app and
// routes /ws, /fusion, /webrtc, /detect... to the detector under one origin, so
// phones only ever see one HTTPS URL and there's no mixed-content / cross-origin
// issue. Built in via NEXT_PUBLIC_BACKEND_SAME_ORIGIN=1 (see docker-compose.yml).
const SAME_ORIGIN =
  process.env.NEXT_PUBLIC_BACKEND_SAME_ORIGIN === "1" ||
  process.env.NEXT_PUBLIC_BACKEND_SAME_ORIGIN === "true";

function backendHost(): string {
  if (process.env.NEXT_PUBLIC_BACKEND_HOST) return process.env.NEXT_PUBLIC_BACKEND_HOST;
  // window.location.host includes the port (or none, behind the tunnel's 443).
  if (SAME_ORIGIN && typeof window !== "undefined") return window.location.host;
  if (typeof window !== "undefined") return `${window.location.hostname}:${BACKEND_PORT}`;
  return `localhost:${BACKEND_PORT}`;
}

export function httpBase(): string {
  const proto = typeof window !== "undefined" && window.location.protocol === "https:" ? "https" : "http";
  return `${proto}://${backendHost()}`;
}

export function wsBase(): string {
  const proto = typeof window !== "undefined" && window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${backendHost()}`;
}
