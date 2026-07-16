import { useEffect, useState } from "react";
import { AlertTriangle, Download, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { BeforeAfterSlider } from "@/components/BeforeAfterSlider";
import { fetchResultObjectUrl } from "@/lib/api";
import type { TrackedJob } from "@/types";
import { cn } from "@/lib/utils";

function baseName(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(0, dot) : name;
}

function StatusBadge({ job }: { job: TrackedJob }) {
  switch (job.status) {
    case "queued":
      return (
        <Badge tone="neutral">
          Queued
          {job.queuePosition != null && job.queuePosition > 0
            ? ` · #${job.queuePosition}`
            : ""}
        </Badge>
      );
    case "running":
      return (
        <Badge tone="primary">
          <Loader2 className="h-3 w-3 animate-spin" />
          Processing
        </Badge>
      );
    case "done":
      return <Badge tone="success">Done</Badge>;
    case "failed":
      return <Badge tone="danger">Failed</Badge>;
  }
}

export function JobCard({ job }: { job: TrackedJob }) {
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultError, setResultError] = useState<string | null>(null);

  // Once the result exists, pull it as an object URL (works with the API token).
  useEffect(() => {
    if (job.status !== "done" || !job.hasResult || resultUrl) return;
    let revoked = false;
    let created: string | null = null;
    fetchResultObjectUrl(job.jobId)
      .then((url) => {
        if (revoked) {
          URL.revokeObjectURL(url);
          return;
        }
        created = url;
        setResultUrl(url);
      })
      .catch((e: Error) => setResultError(e.message));
    return () => {
      revoked = true;
      if (created) URL.revokeObjectURL(created);
    };
  }, [job.status, job.hasResult, job.jobId, resultUrl]);

  const active = job.status === "queued" || job.status === "running";

  return (
    <div className="animate-fade-up overflow-hidden rounded-lg border border-border bg-card shadow-card">
      <div className="flex items-center justify-between gap-3 px-4 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <span
            className={cn(
              "h-2 w-2 shrink-0 rounded-full",
              job.status === "done" && "bg-success",
              job.status === "failed" && "bg-destructive",
              job.status === "running" && "bg-primary",
              job.status === "queued" && "bg-muted-foreground/50",
            )}
          />
          <p
            dir="auto"
            className="min-w-0 flex-1 truncate text-sm font-medium"
            title={job.name}
          >
            {job.name}
          </p>
        </div>
        <StatusBadge job={job} />
      </div>

      {active && (
        <div className="relative aspect-[4/3] w-full overflow-hidden">
          <img
            src={job.originalUrl}
            alt={job.name}
            className={cn(
              "h-full w-full object-contain",
              job.status === "running" ? "opacity-90" : "opacity-60 grayscale",
            )}
          />
          <div className="shimmer pointer-events-none absolute inset-0" />
          <div className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-gradient-to-t from-black/60 to-transparent px-3 pb-2.5 pt-8 text-xs text-white">
            {job.status === "running" ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Restoring…
              </>
            ) : (
              <>
                Waiting in queue
                {job.queuePosition != null && job.queuePosition > 0
                  ? ` · position ${job.queuePosition}`
                  : ""}
              </>
            )}
          </div>
        </div>
      )}

      {job.status === "failed" && (
        <div className="mx-4 mb-4 flex items-start gap-2.5 rounded-lg bg-destructive/10 px-3 py-2.5 text-sm text-destructive">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span className="min-w-0 break-words">
            {job.error || "Processing failed. Try again with different options."}
          </span>
        </div>
      )}

      {job.status === "done" && (
        <div className="space-y-3 px-4 pb-4">
          {resultUrl ? (
            <BeforeAfterSlider
              beforeUrl={job.originalUrl}
              afterUrl={resultUrl}
            />
          ) : resultError ? (
            <div className="flex items-start gap-2.5 rounded-lg bg-destructive/10 px-3 py-2.5 text-sm text-destructive">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{resultError}</span>
            </div>
          ) : (
            <div className="grid aspect-[4/3] w-full place-items-center rounded-lg bg-muted">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          )}

          {resultUrl && (
            <a
              href={resultUrl}
              download={`${baseName(job.name)}-rechroma.png`}
              className="block"
            >
              <Button variant="outline" size="md" className="w-full">
                <Download className="h-4 w-4" />
                Download result
              </Button>
            </a>
          )}
        </div>
      )}
    </div>
  );
}
