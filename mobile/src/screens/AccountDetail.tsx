import { useParams, Link } from "react-router-dom";
import { ArrowLeft, LineChart, Receipt } from "lucide-react";
import { Screen } from "../components/Screen";
import { Card } from "../components/Card";
import { Pill, directionTone, statusTone } from "../components/Pill";
import { EmptyState, ErrorState, SkeletonCard, SkeletonList } from "../components/States";
import { useAccount, useAccountAllocations } from "../api/hooks";
import { formatMoney } from "../lib/format";
import type { AllocationOverviewResponse } from "../api/types";

function TradeRow({ trade }: { trade: AllocationOverviewResponse }) {
  const risk = trade.applied_risk_pct ?? trade.planned_risk_pct;
  return (
    <Link to={`/trades/${trade.allocation_id}`}>
      <Card className="animate-in transition-transform active:scale-[0.98]" padded>
        <div className="flex items-start justify-between">
          <div>
            <p className="text-lg font-bold text-ink">{trade.pair}</p>
            <div className="mt-2 flex gap-2">
              <Pill tone={directionTone(trade.direction)}>{trade.direction}</Pill>
              <Pill tone={statusTone(trade.status)}>{trade.status}</Pill>
            </div>
          </div>
          <div className="text-right">
            {trade.realized_r !== null ? (
              <>
                <p className="text-xs font-medium text-ink-soft">Result</p>
                <p
                  className={`text-lg font-bold ${
                    Number(trade.realized_r) > 0
                      ? "text-positive"
                      : Number(trade.realized_r) < 0
                        ? "text-negative"
                        : "text-ink-soft"
                  }`}
                >
                  {Number(trade.realized_r) > 0 ? "+" : ""}
                  {Number(trade.realized_r).toFixed(2)}R
                </p>
              </>
            ) : (
              <>
                <p className="text-xs font-medium text-ink-soft">Risk</p>
                <p className="text-lg font-bold text-ink">{Number(risk).toFixed(1)}%</p>
              </>
            )}
          </div>
        </div>
      </Card>
    </Link>
  );
}

export function AccountDetail() {
  const { accountId } = useParams<{ accountId: string }>();
  const account = useAccount(accountId);
  const allocations = useAccountAllocations(accountId);

  if (account.isError) {
    return (
      <Screen title="Account">
        <ErrorState message={(account.error as Error).message} />
      </Screen>
    );
  }

  if (account.isPending || !account.data) {
    return (
      <Screen title="Account">
        <SkeletonCard height={140} />
      </Screen>
    );
  }

  const acc = account.data;
  const openTrades = (allocations.data ?? []).filter((a) => a.status !== "CLOSED");
  const closedTrades = (allocations.data ?? []).filter((a) => a.status === "CLOSED");

  return (
    <Screen title={acc.account_type}>
      <div className="flex flex-col gap-4">
        <Link to="/empire" className="-mt-2 mb-1 flex items-center gap-1 text-sm font-semibold text-ink-soft">
          <ArrowLeft size={16} strokeWidth={1.75} />
          Empire
        </Link>

        <Card className="animate-in text-center">
          <p className="text-sm font-medium text-ink-soft">Balance</p>
          <p className="mt-2 text-[40px] font-extrabold leading-none tracking-tight text-ink">
            {formatMoney(acc.balance, { compact: true })}
          </p>
          <div className="mt-3 flex items-center justify-center gap-2">
            <Pill tone={acc.status === "ACTIVE" ? "positive" : "neutral"}>{acc.status}</Pill>
            {acc.equity !== acc.balance && (
              <span className="text-sm text-ink-soft">Equity {formatMoney(acc.equity, { compact: true })}</span>
            )}
          </div>
        </Card>

        <div>
          <p className="mb-3 text-sm font-semibold text-ink-soft">Offene Trades</p>
          {allocations.isPending ? (
            <SkeletonList count={2} height={110} />
          ) : openTrades.length > 0 ? (
            <div className="flex flex-col gap-3">
              {openTrades.map((trade) => (
                <TradeRow key={trade.allocation_id} trade={trade} />
              ))}
            </div>
          ) : (
            <EmptyState icon={LineChart} title="Keine offenen Trades" />
          )}
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-ink-soft">Geschlossene Trades</p>
          {allocations.isPending ? (
            <SkeletonList count={2} height={110} />
          ) : closedTrades.length > 0 ? (
            <div className="flex flex-col gap-3">
              {closedTrades.map((trade) => (
                <TradeRow key={trade.allocation_id} trade={trade} />
              ))}
            </div>
          ) : (
            <EmptyState icon={LineChart} title="Noch keine geschlossenen Trades" />
          )}
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-ink-soft">Payouts</p>
          <EmptyState icon={Receipt} title="Noch keine Auszahlungen erfasst" />
        </div>
      </div>
    </Screen>
  );
}
