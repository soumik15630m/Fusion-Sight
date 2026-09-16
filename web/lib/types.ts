// Mirrors detector/app/schemas.py -- keep in sync by hand, this is the only
// place the wire contract is duplicated on the client.

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  track_id: number | null;
  moving: boolean | null;
  // Fusion fields -- absent/false unless FUSION_ENABLED and this feed has a
  // pose. See fusion/engine.py.
  is_ghost: boolean;
  source_feed: string | null;
  world_lat: number | null;
  world_lon: number | null;
  bearing_deg: number | null;
  in_frame: boolean | null;
}

export interface DetectionResponse {
  source_id: string;
  frame_width: number;
  frame_height: number;
  inference_ms: number;
  detections: Detection[];
  error?: string;
}

/** Compact per-frame status the detector sends back to the phone on /ws/track.
 * The phone displays only ghosts (from the fusion service on /ws/ghosts), so
 * the full detection list isn't sent down this path -- just this ack. */
export interface TrackAck {
  source_id: string;
  inference_ms: number;
  own_count: number;
  frame_width: number;
  frame_height: number;
  error?: string;
}

export interface PoseUpdate {
  lat: number;
  lon: number;
  alt?: number;
  heading_deg?: number;
  tilt_deg?: number;
  accuracy_m?: number;
  fov_deg?: number;
}

export interface FusionFeed {
  lat: number;
  lon: number;
  alt: number;
  heading_deg: number;
  accuracy_m: number | null;
  name: string | null;
  ts_ms: number;
}

export type FusionFeeds = Record<string, FusionFeed>;

/** One deduplicated real-world object on the operator's unified map --
 * GET /fusion/detections (detector/app/routers/pose.py). `confirmations` is
 * how many distinct feeds saw it: 1 = single-feed sighting, 2+ = cross-confirmed. */
export interface WorldObject {
  lat: number;
  lon: number;
  class_name: string;
  confirmations: number;
  sources: string[];
}
