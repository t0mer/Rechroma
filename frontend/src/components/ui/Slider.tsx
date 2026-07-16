import { cn } from "@/lib/utils";

export interface SliderProps {
  min: number;
  max: number;
  step?: number;
  value: number;
  onValueChange: (value: number) => void;
  disabled?: boolean;
  id?: string;
  "aria-label"?: string;
}

/**
 * A range slider whose filled track and thumb are driven by the current value.
 * Built on a native <input type="range"> for keyboard + a11y, restyled to match.
 */
export function Slider({
  min,
  max,
  step = 1,
  value,
  onValueChange,
  disabled,
  id,
  ...aria
}: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className={cn("relative h-6 w-full", disabled && "opacity-50")}>
      <div className="absolute top-1/2 h-1.5 w-full -translate-y-1/2 rounded-full bg-input" />
      <div
        className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-primary"
        style={{ width: `${pct}%` }}
      />
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onValueChange(Number(e.target.value))}
        className={cn(
          "slider-input absolute inset-0 h-6 w-full cursor-pointer appearance-none bg-transparent",
          "focus-visible:outline-none",
        )}
        {...aria}
      />
      <style>{`
        .slider-input::-webkit-slider-thumb {
          -webkit-appearance: none;
          height: 1.15rem;
          width: 1.15rem;
          border-radius: 9999px;
          background: hsl(var(--card));
          border: 2px solid hsl(var(--primary));
          box-shadow: 0 1px 3px hsl(var(--shadow) / 0.3);
          cursor: pointer;
          transition: transform 0.12s ease;
        }
        .slider-input:focus-visible::-webkit-slider-thumb {
          box-shadow: 0 0 0 4px hsl(var(--ring) / 0.3);
        }
        .slider-input:active::-webkit-slider-thumb { transform: scale(1.12); }
        .slider-input::-moz-range-thumb {
          height: 1.15rem;
          width: 1.15rem;
          border-radius: 9999px;
          background: hsl(var(--card));
          border: 2px solid hsl(var(--primary));
          box-shadow: 0 1px 3px hsl(var(--shadow) / 0.3);
          cursor: pointer;
        }
      `}</style>
    </div>
  );
}
