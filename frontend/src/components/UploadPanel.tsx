import { useCallback, useRef, useState } from "react";
import { ImagePlus, UploadCloud, X } from "lucide-react";
import { cn, formatBytes } from "@/lib/utils";

export interface PendingFile {
  id: string;
  file: File;
  previewUrl: string;
}

const ACCEPT = ".jpg,.jpeg,.png,.webp,.tif,.tiff,.bmp";
const ACCEPTED_MIME = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/tiff",
  "image/bmp",
]);
const ACCEPTED_EXT = /\.(jpe?g|png|webp|tiff?|bmp)$/i;

function isAccepted(file: File): boolean {
  return ACCEPTED_MIME.has(file.type) || ACCEPTED_EXT.test(file.name);
}

interface UploadPanelProps {
  files: PendingFile[];
  onAdd: (files: File[]) => void;
  onRemove: (id: string) => void;
  onRejected: (names: string[]) => void;
}

export function UploadPanel({
  files,
  onAdd,
  onRemove,
  onRejected,
}: UploadPanelProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (list: FileList | null) => {
      if (!list) return;
      const incoming = Array.from(list);
      const ok = incoming.filter(isAccepted);
      const bad = incoming.filter((f) => !isAccepted(f));
      if (ok.length) onAdd(ok);
      if (bad.length) onRejected(bad.map((f) => f.name));
    },
    [onAdd, onRejected],
  );

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        className={cn(
          "group relative grid cursor-pointer place-items-center rounded-xl border-2 border-dashed px-6 py-9 text-center",
          "transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          dragging
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/50 hover:bg-muted/30",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          multiple
          className="hidden"
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <div
          className={cn(
            "mb-3 grid h-12 w-12 place-items-center rounded-full transition-colors",
            dragging
              ? "bg-primary/15 text-primary"
              : "bg-muted text-muted-foreground group-hover:text-primary",
          )}
        >
          <UploadCloud className="h-6 w-6" />
        </div>
        <p className="text-sm font-medium text-foreground">
          Drop photos here, or click to browse
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          JPEG, PNG, WebP, TIFF or BMP · multiple at once
        </p>
      </div>

      {files.length > 0 && (
        <div className="grid grid-cols-3 gap-2.5 sm:grid-cols-4">
          {files.map((f) => (
            <figure
              key={f.id}
              className="group relative aspect-square overflow-hidden rounded-lg border border-border bg-muted"
            >
              <img
                src={f.previewUrl}
                alt={f.file.name}
                loading="lazy"
                className="h-full w-full object-cover"
                onError={(e) => {
                  (e.currentTarget.style.display = "none");
                }}
              />
              <ImagePlus className="absolute left-1/2 top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 text-muted-foreground/50" />
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove(f.id);
                }}
                className="absolute right-1 top-1 grid h-6 w-6 place-items-center rounded-full bg-background/85 text-foreground opacity-0 shadow-sm backdrop-blur transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
                aria-label={`Remove ${f.file.name}`}
              >
                <X className="h-3.5 w-3.5" />
              </button>
              <figcaption className="absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-black/70 to-transparent px-1.5 pb-1 pt-4 text-[0.65rem] text-white">
                <span dir="auto" className="block truncate">
                  {f.file.name}
                </span>
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      {files.length > 0 && (
        <p className="text-xs text-muted-foreground">
          {files.length} photo{files.length > 1 ? "s" : ""} ready ·{" "}
          {formatBytes(files.reduce((n, f) => n + f.file.size, 0))}
        </p>
      )}
    </div>
  );
}
