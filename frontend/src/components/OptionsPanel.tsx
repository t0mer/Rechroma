import { useEffect, useState, type ReactNode } from "react";
import { AlertTriangle, Cpu, ShieldAlert, Sparkles, Zap } from "lucide-react";
import { Segmented, Select } from "@/components/ui/Select";
import { Slider } from "@/components/ui/Slider";
import { Switch } from "@/components/ui/Switch";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";
import {
  listEngines,
  type AnimateEngine,
  type ColorizerModel,
  type EngineInfo,
  type Preset,
} from "@/lib/api";

export interface Options {
  preset: Preset;
  model: ColorizerModel;
  autoRenderFactor: boolean;
  renderFactor: number;
  upscale: "off" | "2" | "4";
  restoreFaces: boolean;
  engine: AnimateEngine;
}

export const DEFAULT_OPTIONS: Options = {
  preset: "full",
  model: "artistic",
  autoRenderFactor: true,
  renderFactor: 30,
  upscale: "2",
  restoreFaces: true,
  engine: "tpsmm",
};

interface OptionsPanelProps {
  options: Options;
  onChange: (patch: Partial<Options>) => void;
}

function Field({
  label,
  hint,
  htmlFor,
  children,
}: {
  label: string;
  hint?: ReactNode;
  htmlFor?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <label
          htmlFor={htmlFor}
          className="text-sm font-medium text-foreground"
        >
          {label}
        </label>
        {hint}
      </div>
      {children}
    </div>
  );
}

function EngineIcon({ engine }: { engine: AnimateEngine }) {
  if (engine === "diffusion") return <Zap className="h-4 w-4 text-accent" />;
  if (engine === "cloud") return <ShieldAlert className="h-4 w-4 text-accent" />;
  return <Cpu className="h-4 w-4 text-accent" />;
}

function EngineSelector({
  value,
  onChange,
}: {
  value: AnimateEngine;
  onChange: (engine: AnimateEngine) => void;
}) {
  const [engines, setEngines] = useState<EngineInfo[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let alive = true;
    listEngines()
      .then((e) => alive && setEngines(e))
      .catch(() => alive && setError(true));
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <p className="text-xs text-muted-foreground">
        Couldn’t load engines; the default will be used.
      </p>
    );
  }
  if (!engines) {
    return (
      <p className="text-xs text-muted-foreground">Loading engines…</p>
    );
  }

  return (
    <div className="space-y-2">
      {engines.map((e) => {
        const selected = value === e.name;
        const disabled = !e.available;
        return (
          <button
            key={e.name}
            type="button"
            disabled={disabled}
            onClick={() => onChange(e.name)}
            className={cn(
              "flex w-full items-start gap-2.5 rounded-lg border px-3 py-2.5 text-left transition-colors",
              selected && !disabled
                ? "border-primary bg-primary/5"
                : "border-border hover:bg-muted/40",
              disabled && "cursor-not-allowed opacity-60 hover:bg-transparent",
            )}
          >
            <span className="mt-0.5">
              <EngineIcon engine={e.name} />
            </span>
            <span className="min-w-0 flex-1 leading-tight">
              <span className="flex flex-wrap items-center gap-1.5">
                <span className="text-sm font-medium">{e.label}</span>
                {e.requires_gpu && <Badge tone="neutral">GPU</Badge>}
                {e.requires_key && <Badge tone="neutral">Key</Badge>}
                {e.name === "cloud" && <Badge tone="warning">Sends photo out</Badge>}
              </span>
              <span className="mt-0.5 block text-xs text-muted-foreground">
                {e.notes}
              </span>
              {disabled && e.reason && (
                <span className="mt-1 flex items-center gap-1 text-xs text-destructive">
                  <AlertTriangle className="h-3 w-3 shrink-0" />
                  {e.reason}
                </span>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function OptionsPanel({ options, onChange }: OptionsPanelProps) {
  const colorizeEnabled = options.preset !== "restore";

  return (
    <div className="space-y-5">
      <Field label="Preset">
        <Segmented<Preset>
          value={options.preset}
          onValueChange={(v) => onChange({ preset: v })}
          options={[
            { value: "colorize", label: "Colorize" },
            { value: "restore", label: "Restore" },
            { value: "full", label: "Full" },
            { value: "animate", label: "Animate" },
          ]}
        />
        <p className="text-xs text-muted-foreground">
          {options.preset === "colorize" &&
            "Add colour to a black & white photo."}
          {options.preset === "restore" &&
            "Repair faces, denoise and sharpen an already-colour photo."}
          {options.preset === "full" &&
            "Restore, then colourise — the full revival."}
          {options.preset === "animate" &&
            "Bring the face to life — a short animated clip from a still portrait. " +
              "Best on a clear, front-facing photo; slow on CPU."}
        </p>
      </Field>

      {options.preset === "animate" ? (
        <>
          <div className="h-px bg-border" />
          <Field label="Engine">
            <EngineSelector
              value={options.engine}
              onChange={(engine) => onChange({ engine })}
            />
          </Field>
        </>
      ) : (
      <>
      <div className="h-px bg-border" />

      <Field
        label="Colorizer model"
        hint={
          !colorizeEnabled ? (
            <Badge tone="neutral">not used</Badge>
          ) : undefined
        }
      >
        <Select
          value={options.model}
          disabled={!colorizeEnabled}
          onValueChange={(v) => onChange({ model: v as ColorizerModel })}
          options={[
            { value: "artistic", label: "Artistic — vivid, bold colour" },
            { value: "stable", label: "Stable — natural, portraits" },
          ]}
        />
      </Field>

      <Field
        label="Render factor"
        htmlFor="render-factor"
        hint={
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Auto</span>
            <Switch
              checked={options.autoRenderFactor}
              onCheckedChange={(v) => onChange({ autoRenderFactor: v })}
              label="Use device default render factor"
            />
          </div>
        }
      >
        {options.autoRenderFactor ? (
          <p className="rounded-lg bg-muted/50 px-3 py-2.5 text-xs text-muted-foreground">
            Using the device default (higher on GPU, lower on CPU).
          </p>
        ) : (
          <div className="flex items-center gap-3">
            <Slider
              id="render-factor"
              min={7}
              max={45}
              value={options.renderFactor}
              onValueChange={(v) => onChange({ renderFactor: v })}
              aria-label="Render factor"
            />
            <span className="w-8 shrink-0 text-right font-display text-sm font-semibold tabular-nums">
              {options.renderFactor}
            </span>
          </div>
        )}
        {!options.autoRenderFactor && (
          <p className="text-xs text-muted-foreground">
            Higher is more detailed but slower. 7–45.
          </p>
        )}
      </Field>

      <div className="h-px bg-border" />

      <Field label="Upscale">
        <Segmented<"off" | "2" | "4">
          value={options.upscale}
          onValueChange={(v) => onChange({ upscale: v })}
          options={[
            { value: "off", label: "Off" },
            { value: "2", label: "2×" },
            { value: "4", label: "4×" },
          ]}
        />
      </Field>

      <div className="flex items-center justify-between gap-3 rounded-lg bg-muted/40 px-3.5 py-3">
        <div className="flex items-center gap-2.5">
          <Sparkles className="h-4 w-4 text-accent" />
          <div className="leading-tight">
            <p className="text-sm font-medium">Restore faces</p>
            <p className="text-xs text-muted-foreground">
              Sharpen and repair faces (GFPGAN)
            </p>
          </div>
        </div>
        <Switch
          checked={options.restoreFaces}
          onCheckedChange={(v) => onChange({ restoreFaces: v })}
          label="Restore faces"
        />
      </div>
      </>
      )}
    </div>
  );
}
