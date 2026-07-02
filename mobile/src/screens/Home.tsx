import { Link } from "react-router-dom";
import { LineChart, Newspaper, ArrowRight, FileText } from "lucide-react";
import { Screen } from "../components/Screen";
import { Card } from "../components/Card";
import { Pill } from "../components/Pill";
import { SkeletonCard, SkeletonList, ErrorState } from "../components/States";
import { usePortfolio, useActiveTrades, useWeeklyReports, useMarketSnapshot } from "../api/hooks";
import { CURRENT_USER_NAME } from "../api/client";
import { formatMoney, formatSignedPercent, formatWeekRange, isoWeekNumber, formatDate } from "../lib/format";

const NARRATIVE_PREVIEW_LENGTH = 280;

function greeting(): string {
  const hour = new Date().getHours();
  if (hour < 5) return "Noch wach";
  if (hour < 11) return "Guten Morgen";
  if (hour < 18) return "Guten Tag";
  return "Guten Abend";
}

export function Home() {
  const portfolio = usePortfolio();
  const activeTrades = useActiveTrades();
  const reports = useWeeklyReports();
  const marketSnapshot = useMarketSnapshot();

  if (portfolio.isError) {
    return (
      <Screen title={`${greeting()}, ${CURRENT_USER_NAME}`}>
        <ErrorState message={(portfolio.error as Error).message} />
      </Screen>
    );
  }

  const balance = portfolio.data ? Number(portfolio.data.total_balance) : 0;
  const equity = portfolio.data ? Number(portfolio.data.total_equity) : 0;
  const floatingPct = balance !== 0 ? ((equity - balance) / balance) * 100 : 0;
  const deltaTone = floatingPct > 0 ? "text-positive" : floatingPct < 0 ? "text-negative" : "text-ink-soft";

  const focusPairs = Array.from(new Set((activeTrades.data ?? []).map((t) => t.pair))).slice(0, 6);
  const publishedReports = (reports.data ?? []).filter((r) => r.status === "PUBLISHED");
  const latestReport = publishedReports[0];

  return (
    <Screen title={`${greeting()}, ${CURRENT_USER_NAME}`}>
      <div className="flex flex-col gap-4">
        {marketSnapshot.data && (
          <div className="-mt-2 flex justify-end">
            <Pill tone="neutral">Regime: {marketSnapshot.data.regime}</Pill>
          </div>
        )}

        <Card className="animate-in text-center">
          {portfolio.isPending ? (
            <SkeletonCard height={120} />
          ) : (
            <>
              <p className="text-sm font-medium text-ink-soft">Total Capital</p>
              <p className="mt-2 text-[44px] font-extrabold leading-none tracking-tight text-ink">
                {formatMoney(balance, { compact: true })}
              </p>
              {equity !== balance && (
                <p className={`mt-2 text-sm font-semibold ${deltaTone}`}>
                  {formatSignedPercent(floatingPct)} floating
                </p>
              )}
            </>
          )}
        </Card>

        <div className="grid grid-cols-2 gap-4">
          <Link to="/trades">
            <Card className="animate-in transition-transform active:scale-[0.98]">
              <LineChart size={18} strokeWidth={1.75} className="text-primary" />
              <p className="mt-3 text-2xl font-bold text-ink">
                {portfolio.isPending ? "—" : portfolio.data?.active_trade_count}
              </p>
              <p className="text-sm text-ink-soft">Active Trades</p>
            </Card>
          </Link>
          <Link to="/reports">
            <Card className="animate-in transition-transform active:scale-[0.98]">
              <Newspaper size={18} strokeWidth={1.75} className="text-primary" />
              <p className="mt-3 text-2xl font-bold text-ink">
                {reports.isPending ? "—" : publishedReports.length}
              </p>
              <p className="text-sm text-ink-soft">Weekly Reports</p>
            </Card>
          </Link>
        </div>

        {activeTrades.isPending ? (
          <SkeletonCard height={64} />
        ) : (
          focusPairs.length > 0 && (
            <Card className="animate-in">
              <p className="text-sm font-semibold text-ink-soft">Today's Focus</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {focusPairs.map((pair) => (
                  <span
                    key={pair}
                    className="rounded-pill border border-hairline px-3 py-1.5 text-sm font-semibold text-ink"
                  >
                    {pair}
                  </span>
                ))}
              </div>
            </Card>
          )
        )}

        <div>
          <p className="mb-3 text-sm font-semibold text-ink-soft">Weekly Narrative</p>
          {reports.isPending ? (
            <SkeletonList count={1} height={140} />
          ) : latestReport ? (
            <Link to="/reports">
              <Card className="animate-in transition-transform active:scale-[0.98]">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-primary">
                      Week {isoWeekNumber(latestReport.period_start)}
                    </p>
                    <p className="mt-1 text-lg font-bold text-ink">
                      {formatWeekRange(latestReport.period_start, latestReport.period_end)}
                    </p>
                  </div>
                  <ArrowRight size={20} strokeWidth={1.75} className="mt-1 shrink-0 text-ink-soft" />
                </div>
                {latestReport.summary ? (
                  <p className="mt-3 border-t border-hairline pt-3 text-sm leading-relaxed text-ink">
                    {latestReport.summary.length > NARRATIVE_PREVIEW_LENGTH
                      ? `${latestReport.summary.slice(0, NARRATIVE_PREVIEW_LENGTH).trimEnd()}…`
                      : latestReport.summary}
                  </p>
                ) : (
                  <p className="mt-3 border-t border-hairline pt-3 text-sm text-ink-soft">
                    Noch keine Zusammenfassung hinterlegt.
                  </p>
                )}
                <p className="mt-3 text-sm text-ink-soft">
                  Published {latestReport.published_at ? formatDate(latestReport.published_at) : "—"}
                </p>
              </Card>
            </Link>
          ) : (
            <Card className="animate-in flex items-center gap-3 text-ink-soft">
              <FileText size={20} strokeWidth={1.5} />
              <p className="text-sm">Noch kein veröffentlichter Report</p>
            </Card>
          )}
        </div>
      </div>
    </Screen>
  );
}
