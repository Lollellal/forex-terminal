import type { ReactNode } from "react";

type Tone = "primary" | "positive" | "warning" | "negative" | "neutral";

const TONE_CLASSES: Record<Tone, string> = {
  primary: "bg-primary/10 text-primary",
  positive: "bg-positive/10 text-positive",
  warning: "bg-warning/10 text-warning",
  negative: "bg-negative/10 text-negative",
  neutral: "bg-ink/5 text-ink-soft",
};

export function Pill({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-pill px-2.5 py-1 text-xs font-semibold tracking-wide ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  );
}

export function directionTone(direction: "LONG" | "SHORT"): Tone {
  return direction === "LONG" ? "positive" : "negative";
}

export function statusTone(status: string): Tone {
  if (status === "OPEN") return "primary";
  if (status === "CONFIRMED") return "warning";
  if (status === "CLOSED") return "neutral";
  return "neutral";
}
