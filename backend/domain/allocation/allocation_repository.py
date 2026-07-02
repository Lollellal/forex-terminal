"""TradeAllocation-Repository: Event-Append + State-Tabellen-Update in einer
Transaktion (Hybrid-Ansatz, BACKEND_ARCHITECTURE.md §2.1)."""

from __future__ import annotations

import uuid

from sqlalchemy import Connection, text

from backend.domain.shared.event_envelope import EventEnvelope
from backend.domain.shared.event_store_protocol import EventStoreProtocol
from backend.domain.shared.repository import EventSourcedRepository

from .trade_allocation import TradeAllocation

_UPSERT_SQL = text(
    """
    INSERT INTO core.trade_allocations
        (id, account_id, signal_id, pair, direction, status, planned_risk_pct,
         applied_risk_pct, entry_price_planned, sl_price, tp_price,
         opened_at, closed_at, close_reason, realized_r, version, updated_at)
    VALUES
        (:id, :account_id, :signal_id, :pair, :direction, :status, :planned_risk_pct,
         :applied_risk_pct, :entry_price_planned, :sl_price, :tp_price,
         :opened_at, :closed_at, :close_reason, :realized_r, :version, now())
    ON CONFLICT (id) DO UPDATE SET
        status               = EXCLUDED.status,
        applied_risk_pct     = EXCLUDED.applied_risk_pct,
        opened_at            = EXCLUDED.opened_at,
        closed_at            = EXCLUDED.closed_at,
        close_reason         = EXCLUDED.close_reason,
        realized_r           = EXCLUDED.realized_r,
        version              = EXCLUDED.version,
        updated_at           = now()
    """
)


class AllocationRepository(EventSourcedRepository[TradeAllocation]):
    def __init__(self, event_store: EventStoreProtocol) -> None:
        super().__init__(event_store)

    @property
    def aggregate_type(self) -> str:
        return "TradeAllocation"

    def _blank_instance(self, aggregate_id: uuid.UUID) -> TradeAllocation:
        return TradeAllocation(aggregate_id)

    def save(self, conn: Connection, aggregate: TradeAllocation) -> list[EventEnvelope]:
        events = super().save(conn, aggregate)
        if events:
            conn.execute(
                _UPSERT_SQL,
                {
                    "id": str(aggregate.id),
                    "account_id": str(aggregate.account_id),
                    "signal_id": str(aggregate.signal_id) if aggregate.signal_id else None,
                    "pair": aggregate.pair,
                    "direction": aggregate.direction,
                    "status": aggregate.status,
                    "planned_risk_pct": aggregate.planned_risk_pct,
                    "applied_risk_pct": aggregate.applied_risk_pct,
                    "entry_price_planned": aggregate.entry_price_planned,
                    "sl_price": aggregate.sl_price,
                    "tp_price": aggregate.tp_price,
                    "opened_at": aggregate.opened_at,
                    "closed_at": aggregate.closed_at,
                    "close_reason": aggregate.close_reason,
                    "realized_r": aggregate.realized_r,
                    "version": aggregate.version,
                },
            )
        return events
