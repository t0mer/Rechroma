import * as React from "react";
import { cn } from "@/lib/utils";

type Tone = "neutral" | "primary" | "accent" | "success" | "danger" | "warning";

const tones: Record<Tone, string> = {
  neutral: "bg-muted text-muted-foreground",
  primary: "bg-primary/12 text-primary",
  accent: "bg-accent/12 text-accent",
  success: "bg-success/15 text-success",
  danger: "bg-destructive/15 text-destructive",
  warning: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
};

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
}

export function Badge({ className, tone = "neutral", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5",
        "text-xs font-medium tracking-tight",
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}
