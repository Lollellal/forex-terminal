import { LineChart, Quote } from "lucide-react";
import { Link } from "react-router-dom";
import { Screen } from "../components/Screen";
import { Card } from "../components/Card";
import { Pill, directionTone, statusTone } from "../components/Pill";
import { EmptyState, ErrorState, SkeletonList } from "../components/States";
import { useActiveTrades, useJournal } from "../api/hooks";

export function ActiveTrades() {
  const trades = useActiveTrades();
  const journal = useJournal();

  if (trades.isError) {
    return (
      <Screen title="Active Trades">
        <ErrorState message={(trades.error as Error).message} />
      </Screen>
    );
  }

  const notesByAllocation = new Map(
    (journal.data ?? []).map((entry) => [
      entry.allocation_id,
      entry.notes.length > 0 ? entry.notes[entry.notes.length - 1].text : null,
    ]),
  );

  return (
    <Screen title="Active Trades" subtitle={trades.data ? `${trades.data.length} open` : undefined}>
      {trades.isPending ? (
        <SkeletonList count={3} height={148} />
      ) : trades.data && trades.data.length > 0 ? (
        <div className="flex flex-col gap-4">
          {trades.data.map((trade) => {
            const risk = trade.applied_risk_pct ?? trade.planned_risk_pct;
            const why = notesByAllocation.get(trade.allocation_id);
            return (
              <Link key={trade.allocation_id} to={`/trades/${trade.allocation_id}`}>
                <Card className="animate-in transition-transform active:scale-[0.98]">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-xl font-bold text-ink">{trade.pair}</p>
                      <div className="mt-2 flex gap-2">
                        <Pill tone={directionTone(trade.direction)}>{trade.direction}</Pill>
                        <Pill tone={statusTone(trade.status)}>{trade.status}</Pill>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-xs font-medium text-ink-soft">Risk</p>
                      <p className="text-lg font-bold text-ink">{Number(risk).toFixed(1)}%</p>
                    </div>
                  </div>
                  {why && (
                    <p className="mt-4 flex items-start gap-2 border-t border-hairline pt-3 text-sm italic text-ink-soft">
                      <Quote size={14} strokeWidth={1.75} className="mt-0.5 shrink-0" />
                      {why}
                    </p>
                  )}
                </Card>
              </Link>
            );
          })}
        </div>
      ) : (
        <EmptyState icon={LineChart} title="Keine offenen Trades" hint="Neue Positionen erscheinen hier." />
      )}
    </Screen>
  );
}
