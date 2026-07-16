import type { JobStatus } from "@/lib/api";

/** A submitted job plus the local original preview, tracked in the UI. */
export interface TrackedJob {
  key: string;
  name: string;
  originalUrl: string;
  jobId: string;
  status: JobStatus;
  queuePosition: number | null;
  error: string | null;
  hasResult: boolean;
}
