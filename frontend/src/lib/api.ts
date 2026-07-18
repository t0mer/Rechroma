/**
 * Thin client over the Rechroma REST API (served same-origin).
 *
 * An optional shared token is read from localStorage and sent as `X-API-Token`
 * so the UI keeps working whether or not `WEB_AUTH_TOKEN` is configured.
 */

export type JobStatus = "queued" | "running" | "done" | "failed";
export type Preset = "colorize" | "restore" | "full" | "animate";
export type ColorizerModel = "artistic" | "stable";
export type AnimateEngine = "tpsmm" | "diffusion" | "cloud";

export type JobKind = "image" | "video" | "animate";

export interface EngineInfo {
  name: AnimateEngine;
  label: string;
  requires_gpu: boolean;
  requires_key: boolean;
  available: boolean;
  reason: string;
  notes: string;
}

export interface Job {
  id: string;
  status: JobStatus;
  kind: JobKind;
  name: string;
  preset: string;
  progress: number;
  queue_position: number | null;
  error: string | null;
  has_result: boolean;
  created_at: number;
}

export interface Health {
  status: string;
  version: string;
  device: string;
  queue_depth: number;
}

export interface JobOptions {
  preset: Preset;
  model: ColorizerModel;
  renderFactor?: number;
  upscale?: 2 | 4;
  restoreFaces: boolean;
  engine?: AnimateEngine;
}

const TOKEN_KEY = "rechroma-token";

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setToken(token: string): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

function authHeaders(): HeadersInit {
  const token = getToken();
  return token ? { "X-API-Token": token } : {};
}

async function extractError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
  } catch {
    /* fall through */
  }
  return `Request failed (${res.status})`;
}

export async function createJob(file: File, options: JobOptions): Promise<Job> {
  const form = new FormData();
  form.append("file", file);
  form.append("preset", options.preset);
  form.append("model", options.model);
  if (options.renderFactor != null) {
    form.append("render_factor", String(options.renderFactor));
  }
  if (options.upscale != null) {
    form.append("upscale", String(options.upscale));
  }
  form.append("restore_faces", options.restoreFaces ? "true" : "false");
  if (options.engine) {
    form.append("engine", options.engine);
  }

  const res = await fetch("/api/v1/jobs", {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) throw new Error(await extractError(res));
  return (await res.json()) as Job;
}

export async function getJob(id: string): Promise<Job> {
  const res = await fetch(`/api/v1/jobs/${id}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(await extractError(res));
  return (await res.json()) as Job;
}

export async function listJobs(): Promise<Job[]> {
  const res = await fetch("/api/v1/jobs", { headers: authHeaders() });
  if (!res.ok) throw new Error(await extractError(res));
  return (await res.json()) as Job[];
}

/** Cancel/remove a job (drop from queue, abort, or dismiss its result). */
export async function deleteJob(id: string): Promise<void> {
  const res = await fetch(`/api/v1/jobs/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok && res.status !== 404) throw new Error(await extractError(res));
}

/** Raw path to the result image. */
export function resultUrl(id: string): string {
  return `/api/v1/jobs/${id}/result`;
}

/**
 * Fetch the result PNG as an object URL. Going through fetch (rather than a bare
 * <img src>) lets us attach the optional API token and works regardless of auth.
 * Caller is responsible for URL.revokeObjectURL when done.
 */
export async function fetchResultObjectUrl(id: string): Promise<string> {
  const res = await fetch(resultUrl(id), { headers: authHeaders() });
  if (!res.ok) throw new Error(await extractError(res));
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export async function getHealth(): Promise<Health> {
  const res = await fetch("/healthz", { headers: authHeaders() });
  if (!res.ok) throw new Error(await extractError(res));
  return (await res.json()) as Health;
}

/** List the animate engines and whether each is usable on this install. */
export async function listEngines(): Promise<EngineInfo[]> {
  const res = await fetch("/api/v1/animate/engines", { headers: authHeaders() });
  if (!res.ok) throw new Error(await extractError(res));
  return (await res.json()) as EngineInfo[];
}
