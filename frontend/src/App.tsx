import { useCallback, useEffect, useRef, useState } from "react";
import { ImageOff, Loader2, Wand2 } from "lucide-react";
import { Header } from "@/components/Header";
import { UploadPanel, type PendingFile } from "@/components/UploadPanel";
import { OptionsPanel, DEFAULT_OPTIONS, type Options } from "@/components/OptionsPanel";
import { JobCard } from "@/components/JobCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useToast } from "@/components/Toaster";
import { useTheme } from "@/hooks/useTheme";
import { useJobPolling } from "@/hooks/useJobPolling";
import { createJob, getHealth, type Health, type Job, type JobOptions } from "@/lib/api";
import type { TrackedJob } from "@/types";

let counter = 0;
const nextKey = () => `k${Date.now()}-${counter++}`;

function toApiOptions(o: Options): JobOptions {
  return {
    preset: o.preset,
    model: o.model,
    renderFactor: o.autoRenderFactor ? undefined : o.renderFactor,
    upscale: o.upscale === "off" ? undefined : (Number(o.upscale) as 2 | 4),
    restoreFaces: o.restoreFaces,
  };
}

export default function App() {
  const { theme, toggle } = useTheme();
  const toast = useToast();

  const [options, setOptions] = useState<Options>(DEFAULT_OPTIONS);
  const [pending, setPending] = useState<PendingFile[]>([]);
  const [tracked, setTracked] = useState<TrackedJob[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);

  const pendingRef = useRef(pending);
  pendingRef.current = pending;

  const patchOptions = useCallback(
    (patch: Partial<Options>) => setOptions((o) => ({ ...o, ...patch })),
    [],
  );

  const addFiles = useCallback((files: File[]) => {
    setPending((prev) => [
      ...prev,
      ...files.map((file) => ({
        id: nextKey(),
        file,
        previewUrl: URL.createObjectURL(file),
      })),
    ]);
  }, []);

  const removePending = useCallback((id: string) => {
    setPending((prev) => {
      const found = prev.find((p) => p.id === id);
      if (found) URL.revokeObjectURL(found.previewUrl);
      return prev.filter((p) => p.id !== id);
    });
  }, []);

  const onRejected = useCallback(
    (names: string[]) =>
      toast.error(
        `Unsupported file${names.length > 1 ? "s" : ""}: ${names.join(", ")}. Use JPEG, PNG, WebP, TIFF or BMP.`,
      ),
    [toast],
  );

  const submit = useCallback(async () => {
    const files = pendingRef.current;
    if (files.length === 0 || submitting) return;
    setSubmitting(true);
    const apiOptions = toApiOptions(options);

    const results = await Promise.allSettled(
      files.map((pf) =>
        createJob(pf.file, apiOptions).then((job) => ({ pf, job })),
      ),
    );

    const newJobs: TrackedJob[] = [];
    let failures = 0;
    for (const r of results) {
      if (r.status === "fulfilled") {
        const { pf, job } = r.value;
        // Transfer ownership of the preview URL to the tracked job (the "before").
        newJobs.push({
          key: pf.id,
          name: pf.file.name,
          originalUrl: pf.previewUrl,
          jobId: job.id,
          status: job.status,
          kind: job.kind,
          progress: job.progress,
          queuePosition: job.queue_position,
          error: job.error,
          hasResult: job.has_result,
        });
      } else {
        failures += 1;
      }
    }

    // Revoke previews of files that failed to submit.
    const acceptedIds = new Set(newJobs.map((j) => j.key));
    files.forEach((pf) => {
      if (!acceptedIds.has(pf.id)) URL.revokeObjectURL(pf.previewUrl);
    });

    setTracked((prev) => [...newJobs, ...prev]);
    setPending([]);
    setSubmitting(false);

    if (failures > 0) {
      const reason =
        results.find((r) => r.status === "rejected") &&
        (results.find((r) => r.status === "rejected") as PromiseRejectedResult)
          .reason;
      toast.error(
        `${failures} photo${failures > 1 ? "s" : ""} couldn't be submitted${
          reason?.message ? `: ${reason.message}` : "."
        }`,
      );
    }
    if (newJobs.length > 0) {
      toast.success(
        `${newJobs.length} photo${newJobs.length > 1 ? "s" : ""} sent for restoration.`,
      );
    }
  }, [options, submitting, toast]);

  // Live polling of active jobs.
  const onUpdate = useCallback((job: Job) => {
    setTracked((prev) =>
      prev.map((t) =>
        t.jobId === job.id
          ? {
              ...t,
              status: job.status,
              kind: job.kind,
              progress: job.progress,
              queuePosition: job.queue_position,
              error: job.error,
              hasResult: job.has_result,
            }
          : t,
      ),
    );
  }, []);

  const onPollError = useCallback((id: string, message: string) => {
    // Transient network hiccups shouldn't fail a job or spam toasts.
    console.warn(`poll ${id}: ${message}`);
  }, []);

  const activeJobs = tracked.filter(
    (t) => t.status === "queued" || t.status === "running",
  );
  const activeIds = activeJobs.map((t) => t.jobId);

  useJobPolling(activeIds, onUpdate, onPollError);

  // Health: on mount and every 30s.
  useEffect(() => {
    let alive = true;
    const load = () =>
      getHealth()
        .then((h) => alive && setHealth(h))
        .catch(() => {});
    load();
    const interval = window.setInterval(load, 30000);
    return () => {
      alive = false;
      window.clearInterval(interval);
    };
  }, []);

  // Revoke all object URLs on unmount.
  useEffect(
    () => () => {
      pendingRef.current.forEach((p) => URL.revokeObjectURL(p.previewUrl));
    },
    [],
  );

  return (
    <div className="min-h-dvh">
      <Header theme={theme} onToggleTheme={toggle} activeJobs={activeJobs} />

      <main className="mx-auto max-w-6xl px-4 pb-16 pt-8 sm:px-6">
        <section className="mb-8 max-w-2xl">
          <h2 className="font-display text-3xl font-bold leading-[1.1] tracking-tight sm:text-4xl">
            Give faded photographs
            <span className="text-primary"> their colour back.</span>
          </h2>
          <p className="mt-3 text-sm text-muted-foreground sm:text-base">
            Upload a scan, choose how far to take it, and watch the restoration
            reveal itself. Nothing is kept once you have your result.
          </p>
        </section>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          {/* Compose column */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Upload</CardTitle>
              </CardHeader>
              <CardContent>
                <UploadPanel
                  files={pending}
                  onAdd={addFiles}
                  onRemove={removePending}
                  onRejected={onRejected}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Options</CardTitle>
              </CardHeader>
              <CardContent>
                <OptionsPanel options={options} onChange={patchOptions} />
              </CardContent>
            </Card>

            <Button
              size="lg"
              className="w-full"
              disabled={pending.length === 0 || submitting}
              onClick={submit}
            >
              {submitting ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Wand2 className="h-5 w-5" />
              )}
              {submitting
                ? "Sending…"
                : pending.length > 0
                  ? `Restore ${pending.length} photo${pending.length > 1 ? "s" : ""}`
                  : "Restore photos"}
            </Button>
          </div>

          {/* Jobs column */}
          <div className="space-y-4">
            <div className="flex items-baseline justify-between">
              <h3 className="font-display text-lg font-semibold">Results</h3>
              {tracked.length > 0 && (
                <span className="text-xs text-muted-foreground">
                  {tracked.length} job{tracked.length > 1 ? "s" : ""}
                </span>
              )}
            </div>

            {tracked.length === 0 ? (
              <div className="grid place-items-center rounded-lg border border-dashed border-border bg-muted/20 px-6 py-16 text-center">
                <div className="mb-3 grid h-12 w-12 place-items-center rounded-full bg-muted text-muted-foreground">
                  <ImageOff className="h-6 w-6" />
                </div>
                <p className="text-sm font-medium">No restorations yet</p>
                <p className="mt-1 max-w-xs text-xs text-muted-foreground">
                  Your finished photos will appear here with a before / after
                  slider.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {tracked.map((job) => (
                  <JobCard key={job.key} job={job} />
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="border-t border-border/70">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-4 text-xs text-muted-foreground sm:px-6">
          <span>Rechroma — self-hosted photo colorization & restoration</span>
          {health && (
            <span className="flex items-center gap-3 font-mono">
              <span>v{health.version}</span>
              <span className="flex items-center gap-1.5">
                <span
                  className={
                    health.device === "cuda"
                      ? "h-1.5 w-1.5 rounded-full bg-success"
                      : "h-1.5 w-1.5 rounded-full bg-muted-foreground/60"
                  }
                />
                {health.device.toUpperCase()}
              </span>
              <span>queue {health.queue_depth}</span>
            </span>
          )}
        </div>
      </footer>
    </div>
  );
}
