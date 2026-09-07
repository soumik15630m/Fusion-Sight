"use client";

import { useEffect, useState } from "react";
import { httpBase } from "@/lib/config";
import type { FusionFeeds } from "@/lib/types";

/** Polls GET /fusion/feeds -- cheap, low-rate (map doesn't need per-frame
 * updates), shared by the device map panel and the operator overview. */
export function useFusionFeeds(intervalMs = 1000): FusionFeeds {
  const [feeds, setFeeds] = useState<FusionFeeds>({});

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const res = await fetch(`${httpBase()}/fusion/feeds`);
        if (!cancelled && res.ok) setFeeds(await res.json());
      } catch {
        // backend unreachable/restarting -- keep last known feeds, try again next tick
      }
    }
    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return feeds;
}
