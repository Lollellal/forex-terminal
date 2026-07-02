import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ArrowLeft, FileText, Newspaper } from "lucide-react";
import { Screen } from "../components/Screen";
import { Card } from "../components/Card";
import { Pill, directionTone, statusTone } from "../components/Pill";
import { JournalCard } from "../components/JournalCard";
import { ErrorState, SkeletonCard } from "../components/States";
import { useAllocationDetail, useJournalEntry } from "../api/hooks";
import { get } from "../api/client";
import type { SignalSnapshot, WeeklyReportDownloadUrlResponse } from "../api/types";
import { formatDate } from "../lib/format";

const SNAPSHOT_FIELDS: { key: keyof SignalSnapshot; label: string }[] = [
  { key: "quality", label: "Quality" },
  { key: "edge", label: "Seasonality Edge" },
  { key: "alignment", label: "Alignment" },
  { key: "combo_key", label: "Combo" },
  { key: "ml_score", label: "ML Score" },
  { key: "overall_score", label: "Overall Score" },
  { key: "seasonality_score", label: "Seasonality Score" },
  { key: "regime", label: "Regime" },
];

function SnapshotRow({ label, value }: { label: string; value: string | number | null }) {
  return (
    <div className="flex items-start justify-between gap-3 py-1.5">
      <span className="shrink-0 text-sm text-ink-soft">{label}</span>
      <span className="text-right text-sm font-semibold text-ink">
        {value === null || value === undefined ? "—" : value}
      </span>
    </div>
  );
}

function WeeklyReportCard({ reportId, reportWeek }: { reportId: string; reportWeek: string | null }) {
  const [opening, setOpening] = useState(false);

  async function open() {
    // Synchron im Klick-Handler öffnen (Popup-Blocker), wie in WeeklyReports.tsx.
    const popup = window.open("", "_blank");
    setOpening(true);
    try {
      const { url } = await get<WeeklyReportDownloadUrlResponse>(`/weekly-reports/${reportId}/download-url`);
      if (popup) popup.location.href = url;
    } finally {
      setOpening(false);
    }
  }

  return (
    <Card className="animate-in">
      <p className="mb-3 text-sm font-semibold text-ink-soft">Report der Woche</p>
      <button
        onClick={open}
        disabled={opening}
        className="flex w-full items-center gap-3 rounded-pill border border-hairline px-3.5 py-2.5 text-left"
      >
        <Newspaper size={18} strokeWidth={1.75} className="shrink-0 text-primary" />
        <span className="text-sm font-semibold text-ink">{reportWeek ?? "Report öffnen"}</span>
      </button>
    </Card>
  );
}

export function TradeDetail() {
  const { allocationId } = useParams<{ allocationId: string }>();
  const allocation = useAllocationDetail(allocationId);
  const journalEntry = useJournalEntry(allocationId);

  if (allocation.isError) {
    return (
      <Screen title="Trade">
        <ErrorState message={(allocation.error as Error).message} />
      </Screen>
    );
  }

  if (allocation.isPending || !allocation.data) {
    return (
      <Screen title="Trade">
        <SkeletonCard height={160} />
      </Screen>
    );
  }

  const trade = allocation.data;
  const risk = trade.applied_risk_pct ?? trade.planned_risk_pct;
  const snapshot = trade.signal_snapshot;

  return (
    <Screen title={trade.pair}>
      <div className="flex flex-col gap-4">
        <Link to="/trades" className="-mt-2 mb-1 flex items-center gap-1 text-sm font-semibold text-ink-soft">
          <ArrowLeft size={16} strokeWidth={1.75} />
          Active Trades
        </Link>

        <Card className="animate-in">
          <div className="flex items-start justify-between">
            <div className="flex gap-2">
              <Pill tone={directionTone(trade.direction)}>{trade.direction}</Pill>
              <Pill tone={statusTone(trade.status)}>{trade.status}</Pill>
            </div>
            <div className="text-right">
              <p className="text-xs font-medium text-ink-soft">Risk</p>
              <p className="text-lg font-bold text-ink">{Number(risk).toFixed(1)}%</p>
            </div>
          </div>
          <div className="mt-4 flex gap-6 border-t border-hairline pt-3 text-sm">
            <div>
              <p className="text-ink-soft">Opened</p>
              <p className="font-semibold text-ink">{trade.opened_at ? formatDate(trade.opened_at) : "—"}</p>
            </div>
            <div>
              <p className="text-ink-soft">Closed</p>
              <p className="font-semibold text-ink">{trade.closed_at ? formatDate(trade.closed_at) : "—"}</p>
            </div>
          </div>
        </Card>

        <Card className="animate-in">
          <p className="mb-1 text-sm font-semibold text-ink-soft">Signal-Kontext</p>
          {snapshot ? (
            <div className="divide-y divide-hairline/60">
              {SNAPSHOT_FIELDS.map((field) => (
                <SnapshotRow key={field.key} label={field.label} value={snapshot[field.key] as string | number | null} />
              ))}
            </div>
          ) : (
            <div className="flex items-center gap-3 py-3 text-ink-soft">
              <FileText size={20} strokeWidth={1.5} />
              <p className="text-sm">Kein Signal-Snapshot vorhanden (vor Einführung des Features importiert)</p>
            </div>
          )}
        </Card>

        {snapshot?.weekly_report_id && (
          <WeeklyReportCard reportId={snapshot.weekly_report_id} reportWeek={snapshot.report_week} />
        )}

        {journalEntry.data && <JournalCard entry={journalEntry.data} />}
      </div>
    </Screen>
  );
}
