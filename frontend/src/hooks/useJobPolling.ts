import { useEffect } from "react";
import { getJob, type Job } from "@/lib/api";

const POLL_MS = 1500;

/**
 * Poll each active job (~1.5s) until it reaches a terminal state. The interval
 * only runs while there is at least one active job, per the live-refresh spec.
 */
export function useJobPolling(
  activeIds: string[],
  onUpdate: (job: Job) => void,
  onError: (id: string, message: string) => void,
): void {
  const key = activeIds.join(",");
  useEffect(() => {
    if (activeIds.length === 0) return;
    let cancelled = false;

    const tick = () => {
      activeIds.forEach(async (id) => {
        try {
          const job = await getJob(id);
          if (!cancelled) onUpdate(job);
        } catch (e) {
          if (!cancelled) onError(id, (e as Error).message);
        }
      });
    };

    const interval = window.setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
    // key captures the set of active ids; callbacks are stable setState wrappers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, onUpdate, onError]);
}
