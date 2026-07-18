import type { ReactNode } from "react";
import { Sparkles } from "lucide-react";
import { Segmented, Select } from "@/components/ui/Select";
import { Slider } from "@/components/ui/Slider";
import { Switch } from "@/components/ui/Switch";
import { Badge } from "@/components/ui/Badge";
import type { ColorizerModel, Preset } from "@/lib/api";

export interface Options {
  preset: Preset;
  model: ColorizerModel;
  autoRenderFactor: boolean;
  renderFactor: number;
  upscale: "off" | "2" | "4";
  restoreFaces: boolean;
}

export const DEFAULT_OPTIONS: Options = {
  preset: "full",
  model: "artistic",
  autoRenderFactor: true,
  renderFactor: 30,
  upscale: "2",
  restoreFaces: true,
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

      {options.preset === "animate" ? null : (
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
