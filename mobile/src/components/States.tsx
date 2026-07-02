import type { LucideIcon } from "lucide-react";
import { AlertTriangle } from "lucide-react";
import { Card } from "./Card";

export function EmptyState({ icon: Icon, title, hint }: { icon: LucideIcon; title: string; hint?: string }) {
  return (
    <Card className="flex flex-col items-center gap-3 py-10 text-center animate-in">
      <Icon size={28} strokeWidth={1.5} className="text-ink-soft" />
      <div>
        <p className="text-sm font-semibold text-ink">{title}</p>
        {hint && <p className="mt-1 text-sm text-ink-soft">{hint}</p>}
      </div>
    </Card>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <Card className="flex flex-col items-center gap-3 py-10 text-center animate-in">
      <AlertTriangle size={28} strokeWidth={1.5} className="text-negative" />
      <div>
        <p className="text-sm font-semibold text-ink">Verbindung fehlgeschlagen</p>
        <p className="mt-1 text-sm text-ink-soft">{message}</p>
      </div>
    </Card>
  );
}

export function SkeletonCard({ height = 88 }: { height?: number }) {
  return (
    <div
      className="animate-pulse rounded-card bg-ink/5"
      style={{ height }}
      aria-hidden
    />
  );
}

export function SkeletonList({ count = 3, height = 88 }: { count?: number; height?: number }) {
  return (
    <div className="flex flex-col gap-3">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} height={height} />
      ))}
    </div>
  );
}
