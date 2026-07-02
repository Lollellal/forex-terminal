"""Account-Repository: Event-Append + State-Tabellen-Update in einer
Transaktion (Hybrid-Ansatz, BACKEND_ARCHITECTURE.md §2.1)."""

from __future__ import annotations

import uuid

from sqlalchemy import Connection, text

from backend.domain.shared.event_envelope import EventEnvelope
from backend.domain.shared.event_store_protocol import EventStoreProtocol
from backend.domain.shared.repository import EventSourcedRepository

from .account import Account

_UPSERT_SQL = text(
    """
    INSERT INTO core.accounts
        (id, user_id, empire_id, account_type, status, balance, equity, version, updated_at)
    VALUES
        (:id, :user_id, :empire_id, :account_type, :status, :balance, :equity, :version, now())
    ON CONFLICT (id) DO UPDATE SET
        empire_id    = EXCLUDED.empire_id,
        status       = EXCLUDED.status,
        balance      = EXCLUDED.balance,
        equity       = EXCLUDED.equity,
        version      = EXCLUDED.version,
        updated_at   = now()
    """
)


class AccountRepository(EventSourcedRepository[Account]):
    def __init__(self, event_store: EventStoreProtocol) -> None:
        super().__init__(event_store)

    @property
    def aggregate_type(self) -> str:
        return "Account"

    def _blank_instance(self, aggregate_id: uuid.UUID) -> Account:
        return Account(aggregate_id)

    def save(self, conn: Connection, aggregate: Account) -> list[EventEnvelope]:
        events = super().save(conn, aggregate)
        if events:
            conn.execute(
                _UPSERT_SQL,
                {
                    "id": str(aggregate.id),
                    "user_id": str(aggregate.user_id),
                    "empire_id": str(aggregate.empire_id) if aggregate.empire_id else None,
                    "account_type": aggregate.account_type,
                    "status": aggregate.status,
                    "balance": aggregate.balance,
                    "equity": aggregate.equity,
                    "version": aggregate.version,
                },
            )
        return events
