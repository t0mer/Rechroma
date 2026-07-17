import { useEffect, useRef, useState } from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import type { TrackedJob } from "@/types";
import { cn } from "@/lib/utils";

/**
 * Header pill + popover that surfaces jobs running in the background so the user
 * always knows work is in flight. Hidden entirely when nothing is active.
 */
export function ActivityIndicator({ jobs }: { jobs: TrackedJob[] }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click / Escape.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const running = jobs.filter((j) => j.status === "running").length;
  if (jobs.length === 0) return null;

  const label = running > 0 ? `${jobs.length} running` : `${jobs.length} queued`;

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label={`${jobs.length} active task${jobs.length > 1 ? "s" : ""}`}
        aria-expanded={open}
        className="flex items-center gap-1.5 rounded-full border border-border bg-card px-2.5 py-1 text-xs font-medium text-foreground shadow-sm transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
        <span className="tabular-nums">{label}</span>
        <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="absolute right-0 top-9 z-40 w-72 animate-fade-up rounded-lg border border-border bg-card p-2 shadow-lift">
          <p className="px-1.5 pb-1.5 pt-0.5 text-xs font-semibold text-muted-foreground">
            Active tasks
          </p>
          <ul className="space-y-1.5">
            {jobs.map((job) => (
              <ActivityRow key={job.jobId} job={job} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ActivityRow({ job }: { job: TrackedJob }) {
  const pct = Math.round(job.progress * 100);
  const showBar = job.status === "running" && job.kind === "video";
  let status: string;
  if (job.status === "queued") {
    status =
      job.queuePosition != null && job.queuePosition > 0
        ? `Queued · #${job.queuePosition}`
        : "Queued";
  } else if (job.kind === "video") {
    status = `Processing · ${pct}%`;
  } else {
    status = "Processing…";
  }

  return (
    <li className="rounded-md px-1.5 py-1">
      <div className="flex items-center justify-between gap-2">
        <span dir="auto" className="min-w-0 flex-1 truncate text-sm" title={job.name}>
          {job.name}
        </span>
        <span className="shrink-0 text-xs text-muted-foreground tabular-nums">{status}</span>
      </div>
      {showBar && (
        <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-[width] duration-500"
            style={{ width: `${Math.max(2, pct)}%` }}
          />
        </div>
      )}
    </li>
  );
}
