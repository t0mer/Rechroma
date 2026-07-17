import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge conditional class names, resolving Tailwind conflicts. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

const VIDEO_EXT = /\.(mp4|mov|webm|mkv|avi)$/i;

/** Whether a file is a video (by MIME type or extension). */
export function isVideoFile(file: File): boolean {
  return file.type.startsWith("video/") || VIDEO_EXT.test(file.name);
}

/**
 * Pick the noun for a batch of files: "video(s)" if all are videos, "photo(s)"
 * if all are photos, "file(s)" for a mix. `count` selects singular/plural.
 */
export function mediaNoun(files: File[], count: number): string {
  const plural = count !== 1;
  if (files.length > 0 && files.every(isVideoFile)) return plural ? "videos" : "video";
  if (files.length > 0 && files.every((f) => !isVideoFile(f))) return plural ? "photos" : "photo";
  return plural ? "files" : "file";
}

/** Human-readable byte size, e.g. 1.4 MB. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`;
}
