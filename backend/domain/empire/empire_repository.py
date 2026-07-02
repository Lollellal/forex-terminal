"""Empire-Repository: Event-Append + State-Tabellen-Update in einer
Transaktion (Hybrid-Ansatz, BACKEND_ARCHITECTURE.md §2.1)."""

from __future__ import annotations

import uuid

from sqlalchemy import Connection, text

from backend.domain.shared.event_envelope import EventEnvelope
from backend.domain.shared.event_store_protocol import EventStoreProtocol
from backend.domain.shared.repository import EventSourcedRepository

from .empire import Empire

_UPSERT_SQL = text(
    """
    INSERT INTO core.empires (id, user_id, name, version, updated_at)
    VALUES (:id, :user_id, :name, :version, now())
    ON CONFLICT (id) DO UPDATE SET
        name       = EXCLUDED.name,
        version    = EXCLUDED.version,
        updated_at = now()
    """
)


class EmpireRepository(EventSourcedRepository[Empire]):
    def __init__(self, event_store: EventStoreProtocol) -> None:
        super().__init__(event_store)

    @property
    def aggregate_type(self) -> str:
        return "Empire"

    def _blank_instance(self, aggregate_id: uuid.UUID) -> Empire:
        return Empire(aggregate_id)

    def save(self, conn: Connection, aggregate: Empire) -> list[EventEnvelope]:
        events = super().save(conn, aggregate)
        if events:
            conn.execute(
                _UPSERT_SQL,
                {
                    "id": str(aggregate.id),
                    "user_id": str(aggregate.user_id),
                    "name": aggregate.name,
                    "version": aggregate.version,
                },
            )
        return events
