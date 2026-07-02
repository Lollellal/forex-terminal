"""Home-Dashboard-Endpunkt. Reine Aggregation über bestehende Projektionen
(account_balances, empire_overview, allocation_overview, journal_view) +
core.accounts.user_id/core.empires.user_id — keine neue Domain-Logik, kein
neuer State. Siehe MOBILE_DATA_CONTRACT.md, Abschnitt "Home": Ziel ist ein
einziger Request für das komplette Dashboard."""

from __future__ import annotations

import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends
from sqlalchemy import Connection, text

from backend.api.dependencies import get_read_conn
from backend.api.schemas.portfolio import (
    EmpireSummary,
    PortfolioResponse,
    RecentJournalEntry,
    StandaloneAccountSummary,
)
from backend.api.schemas.weekly_report import WeeklyReportResponse

router = APIRouter(prefix="/users", tags=["portfolio"])

_EMPIRES_SQL = text(
    """
    SELECT eo.empire_id, eo.name, eo.account_count, eo.total_balance, eo.total_equity
    FROM projections.empire_overview eo
    JOIN core.empires e ON e.id = eo.empire_id
    WHERE e.user_id = :user_id
    ORDER BY eo.total_balance DESC
    """
)
_STANDALONE_ACCOUNTS_SQL = text(
    """
    SELECT ab.account_id, ab.account_type, ab.status, ab.balance, ab.equity
    FROM projections.account_balances ab
    JOIN core.accounts a ON a.id = ab.account_id
    WHERE a.user_id = :user_id AND ab.empire_id IS NULL
    ORDER BY ab.balance DESC
    """
)
_ACTIVE_TRADE_COUNT_SQL = text(
    """
    SELECT COUNT(*) FROM projections.allocation_overview ao
    WHERE ao.account_id IN (SELECT id FROM core.accounts WHERE user_id = :user_id)
      AND ao.status IN ('CONFIRMED','OPEN')
    """
)
_RECENT_JOURNAL_SQL = text(
    """
    SELECT jv.allocation_id, jv.pair, jv.status, jv.updated_at
    FROM projections.journal_view jv
    WHERE jv.account_id IN (SELECT id FROM core.accounts WHERE user_id = :user_id)
    ORDER BY jv.updated_at DESC
    LIMIT :limit
    """
)


@router.get("/{user_id}/portfolio", response_model=PortfolioResponse)
def get_portfolio(
    user_id: uuid.UUID, recent_journal_limit: int = 5, conn: Connection = Depends(get_read_conn)
):
    empire_rows = conn.execute(_EMPIRES_SQL, {"user_id": str(user_id)}).mappings().all()
    standalone_rows = conn.execute(_STANDALONE_ACCOUNTS_SQL, {"user_id": str(user_id)}).mappings().all()
    active_trade_count = conn.execute(_ACTIVE_TRADE_COUNT_SQL, {"user_id": str(user_id)}).scalar_one()
    journal_rows = (
        conn.execute(_RECENT_JOURNAL_SQL, {"user_id": str(user_id), "limit": recent_journal_limit})
        .mappings()
        .all()
    )

    empires = [EmpireSummary(**row) for row in empire_rows]
    standalone_accounts = [StandaloneAccountSummary(**row) for row in standalone_rows]

    total_balance: Decimal = sum((e.total_balance for e in empires), Decimal("0")) + sum(
        (a.balance for a in standalone_accounts), Decimal("0")
    )
    total_equity: Decimal = sum((e.total_equity for e in empires), Decimal("0")) + sum(
        (a.equity for a in standalone_accounts), Decimal("0")
    )

    return PortfolioResponse(
        user_id=user_id,
        total_balance=total_balance,
        total_equity=total_equity,
        empires=empires,
        standalone_accounts=standalone_accounts,
        active_trade_count=active_trade_count,
        recent_journal_entries=[RecentJournalEntry(**row) for row in journal_rows],
    )


_WEEKLY_REPORTS_SQL = text(
    """
    SELECT id, user_id, period_start, period_end, status, content_ref, summary, published_at
    FROM projections.weekly_reports
    WHERE user_id = :user_id
    ORDER BY period_start DESC, updated_at DESC
    LIMIT :limit
    """
)


@router.get("/{user_id}/weekly-reports", response_model=list[WeeklyReportResponse])
def list_weekly_reports(user_id: uuid.UUID, limit: int = 10, conn: Connection = Depends(get_read_conn)):
    """MOBILE_DATA_CONTRACT.md, Abschnitt "Weekly Report" — der in Schritt 7
    spezifizierte, damals nicht gebaute Contract."""
    rows = conn.execute(_WEEKLY_REPORTS_SQL, {"user_id": str(user_id), "limit": limit}).mappings().all()
    return [WeeklyReportResponse(**row) for row in rows]
