import { Card } from "./Card";
import { formatDate } from "../lib/format";
import type { SignalSnapshot } from "../api/types";

// Handelstage (Mo-Fr) zwischen zwei Daten, identisch zur Desktop-/Backend-
// Logik (src/trade_snapshot.py::add_trading_days). Reine Anzeige, keine
// eigene Zeitraum-Quelle -- be_date/close_date kommen bereits fertig
// berechnet aus dem Backend (signal_snapshot).
function tradingDaysBetween(start: Date, end: Date): number {
  let count = 0;
  const d = new Date(start);
  while (d < end) {
    d.setDate(d.getDate() + 1);
    const day = d.getDay();
    if (day !== 0 && day !== 6) count += 1;
  }
  return count;
}

interface TimelineBarProps {
  openedAt: string;
  beDate: string;
  closeDate: string;
  maxHoldDays: number;
  beDays: number;
  compact?: boolean;
}

export function TimelineBar({ openedAt, beDate, closeDate, maxHoldDays, beDays, compact }: TimelineBarProps) {
  const start = new Date(openedAt);
  const today = new Date();
  const elapsed = Math.max(0, Math.min(maxHoldDays, tradingDaysBetween(start, today)));
  const pct = Math.round((elapsed / maxHoldDays) * 100);
  const bePct = Math.round((beDays / maxHoldDays) * 100);
  const beReached = elapsed >= beDays;
  const closeDue = elapsed >= maxHoldDays;

  const bar = (
    <div className="relative h-2 overflow-hidden rounded-pill bg-surface-2">
      <div
        className={`absolute inset-y-0 left-0 rounded-pill transition-all ${beReached ? "bg-positive" : "bg-primary"}`}
        style={{ width: `${pct}%` }}
      />
      <div className="absolute inset-y-[-2px] w-[2px] bg-ink" style={{ left: `${bePct}%` }} />
    </div>
  );

  if (compact) {
    return (
      <div className="mt-3 border-t border-hairline pt-3">
        {bar}
        <div className="mt-1.5 flex justify-between text-[11px] text-ink-soft">
          <span>Tag {elapsed}/{maxHoldDays}</span>
          <span>{beReached ? "BE erreicht" : `BE ${formatDate(beDate)}`}</span>
        </div>
      </div>
    );
  }

  return (
    <Card className="animate-in">
      <p className="mb-3 text-sm font-semibold text-ink-soft">Zeitplan</p>
      {bar}
      <div className="mt-2 flex flex-wrap justify-between gap-x-3 gap-y-1 text-xs text-ink-soft">
        <span>Tag {elapsed} / {maxHoldDays}</span>
        <span>{beReached ? `BE erreicht (${formatDate(beDate)})` : `BE am ${formatDate(beDate)}`}</span>
        <span>{closeDue ? `Zeit-Exit fällig (${formatDate(closeDate)})` : `Spät. Schließen am ${formatDate(closeDate)}`}</span>
      </div>
    </Card>
  );
}

export function WhyCard({ snapshot }: { snapshot: SignalSnapshot }) {
  if (!snapshot.why_base_section && !snapshot.why_quote_section) return null;
  return (
    <Card className="animate-in">
      <p className="mb-1 text-sm font-semibold text-ink-soft">Warum dieser Trade</p>
      <p className="mb-3 text-xs text-ink-soft">Aus {snapshot.report_week}</p>
      <div className="flex flex-col gap-4">
        {snapshot.why_base_section && (
          <div>
            <p className="mb-1 text-xs font-bold uppercase tracking-wide text-ink-soft">{snapshot.why_base_ccy}</p>
            <p className="whitespace-pre-line text-sm leading-relaxed text-ink">{snapshot.why_base_section}</p>
          </div>
        )}
        {snapshot.why_quote_section && (
          <div>
            <p className="mb-1 text-xs font-bold uppercase tracking-wide text-ink-soft">{snapshot.why_quote_ccy}</p>
            <p className="whitespace-pre-line text-sm leading-relaxed text-ink">{snapshot.why_quote_section}</p>
          </div>
        )}
      </div>
    </Card>
  );
}
