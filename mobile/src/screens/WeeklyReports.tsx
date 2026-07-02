import { useState } from "react";
import { ArrowRight, Newspaper } from "lucide-react";
import { Screen } from "../components/Screen";
import { Card } from "../components/Card";
import { Pill } from "../components/Pill";
import { EmptyState, ErrorState, SkeletonList } from "../components/States";
import { useWeeklyReports } from "../api/hooks";
import { get } from "../api/client";
import type { WeeklyReportDownloadUrlResponse, WeeklyReportResponse } from "../api/types";
import { formatDate, formatWeekRange, isoWeekNumber } from "../lib/format";

function ReportCard({ report }: { report: WeeklyReportResponse }) {
  const [opening, setOpening] = useState(false);

  async function open() {
    // Fenster synchron im Klick-Handler öffnen (Popup-Blocker greifen sonst,
    // sobald window.open() erst nach einem await-Fetch aufgerufen wird), die
    // Signed URL wird nachträglich per location.href gesetzt.
    const popup = window.open("", "_blank");
    setOpening(true);
    try {
      const { url } = await get<WeeklyReportDownloadUrlResponse>(
        `/weekly-reports/${report.id}/download-url`,
      );
      if (popup) popup.location.href = url;
    } finally {
      setOpening(false);
    }
  }

  return (
    <button onClick={open} disabled={opening} className="block w-full text-left">
      <Card className="animate-in transition-transform active:scale-[0.98]">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-primary">
              Week {isoWeekNumber(report.period_start)}
            </p>
            <p className="mt-1 text-xl font-bold text-ink">
              {formatWeekRange(report.period_start, report.period_end)}
            </p>
            <div className="mt-2 flex items-center gap-2">
              <Pill tone={report.status === "PUBLISHED" ? "positive" : "neutral"}>{report.status}</Pill>
              {report.published_at && (
                <span className="text-sm text-ink-soft">Published {formatDate(report.published_at)}</span>
              )}
            </div>
          </div>
          <ArrowRight size={20} strokeWidth={1.75} className="mt-1 shrink-0 text-ink-soft" />
        </div>
      </Card>
    </button>
  );
}

export function WeeklyReports() {
  const reports = useWeeklyReports();

  if (reports.isError) {
    return (
      <Screen title="Weekly Reports">
        <ErrorState message={(reports.error as Error).message} />
      </Screen>
    );
  }

  return (
    <Screen title="Weekly Reports">
      {reports.isPending ? (
        <SkeletonList count={3} height={110} />
      ) : reports.data && reports.data.length > 0 ? (
        <div className="flex flex-col gap-4">
          {reports.data.map((report) => (
            <ReportCard key={report.id} report={report} />
          ))}
        </div>
      ) : (
        <EmptyState icon={Newspaper} title="Noch keine Reports" />
      )}
    </Screen>
  );
}
