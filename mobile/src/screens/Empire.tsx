import { Layers } from "lucide-react";
import { Screen } from "../components/Screen";
import { Card } from "../components/Card";
import { Pill } from "../components/Pill";
import { EmptyState, ErrorState, SkeletonCard, SkeletonList } from "../components/States";
import { usePortfolio, useEmpireAccounts } from "../api/hooks";
import { formatMoney } from "../lib/format";
import type { EmpireSummary } from "../api/types";

function EmpireGroup({ empire }: { empire: EmpireSummary }) {
  const accounts = useEmpireAccounts(empire.empire_id);

  return (
    <Card className="animate-in">
      <div className="flex items-baseline justify-between">
        <p className="text-lg font-bold text-ink">{empire.name}</p>
        <p className="text-sm text-ink-soft">{empire.account_count} accounts</p>
      </div>
      <p className="mt-1 text-3xl font-extrabold tracking-tight text-ink">
        {formatMoney(empire.total_balance, { compact: true })}
      </p>

      {accounts.isPending ? (
        <div className="mt-4">
          <SkeletonCard height={48} />
        </div>
      ) : (
        <ul className="mt-4 flex flex-col gap-2 border-t border-hairline pt-3">
          {(accounts.data ?? []).map((account) => (
            <li key={account.account_id} className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2 text-ink-soft">
                {account.account_type}
                <Pill tone={account.status === "ACTIVE" ? "positive" : "neutral"}>{account.status}</Pill>
              </span>
              <span className="font-semibold text-ink">{formatMoney(account.balance)}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export function Empire() {
  const portfolio = usePortfolio();

  if (portfolio.isError) {
    return (
      <Screen title="Empire">
        <ErrorState message={(portfolio.error as Error).message} />
      </Screen>
    );
  }

  if (portfolio.isPending) {
    return (
      <Screen title="Empire">
        <SkeletonList count={2} height={160} />
      </Screen>
    );
  }

  const { empires, standalone_accounts } = portfolio.data!;
  const isEmpty = empires.length === 0 && standalone_accounts.length === 0;

  return (
    <Screen title="Empire" subtitle="Gesamtkapital nach Struktur">
      {isEmpty ? (
        <EmptyState icon={Layers} title="Noch keine Accounts" />
      ) : (
        <div className="flex flex-col gap-4">
          {empires.map((empire) => (
            <EmpireGroup key={empire.empire_id} empire={empire} />
          ))}

          {standalone_accounts.length > 0 && (
            <Card className="animate-in">
              <p className="text-sm font-semibold text-ink-soft">Standalone</p>
              <ul className="mt-3 flex flex-col gap-2.5">
                {standalone_accounts.map((account) => (
                  <li key={account.account_id} className="flex items-center justify-between">
                    <span className="flex items-center gap-2 text-sm text-ink-soft">
                      {account.account_type}
                      <Pill tone={account.status === "ACTIVE" ? "positive" : "neutral"}>
                        {account.status}
                      </Pill>
                    </span>
                    <span className="font-semibold text-ink">{formatMoney(account.balance)}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}
    </Screen>
  );
}
