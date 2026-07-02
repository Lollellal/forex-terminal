"""JournalNote-Repository: Event-Append + State-Tabellen-Update in einer
Transaktion (Hybrid-Ansatz, BACKEND_ARCHITECTURE.md §2.1)."""

from __future__ import annotations

import json
import uuid

from sqlalchemy import Connection, text

from backend.domain.shared.event_envelope import EventEnvelope
from backend.domain.shared.event_store_protocol import EventStoreProtocol
from backend.domain.shared.repository import EventSourcedRepository

from .journal_note import JournalNote

_UPSERT_SQL = text(
    """
    INSERT INTO core.journal_notes
        (id, related_allocation_id, related_signal_id, text, attachments, version, edited_at)
    VALUES
        (:id, :related_allocation_id, :related_signal_id, :text, CAST(:attachments AS jsonb),
         :version, :edited_at)
    ON CONFLICT (id) DO UPDATE SET
        text        = EXCLUDED.text,
        attachments = EXCLUDED.attachments,
        version     = EXCLUDED.version,
        edited_at   = EXCLUDED.edited_at
    """
)


class JournalNoteRepository(EventSourcedRepository[JournalNote]):
    def __init__(self, event_store: EventStoreProtocol) -> None:
        super().__init__(event_store)

    @property
    def aggregate_type(self) -> str:
        return "JournalNote"

    def _blank_instance(self, aggregate_id: uuid.UUID) -> JournalNote:
        return JournalNote(aggregate_id)

    def save(self, conn: Connection, aggregate: JournalNote) -> list[EventEnvelope]:
        events = super().save(conn, aggregate)
        if events:
            conn.execute(
                _UPSERT_SQL,
                {
                    "id": str(aggregate.id),
                    "related_allocation_id": (
                        str(aggregate.related_allocation_id) if aggregate.related_allocation_id else None
                    ),
                    "related_signal_id": (
                        str(aggregate.related_signal_id) if aggregate.related_signal_id else None
                    ),
                    "text": aggregate.text,
                    "attachments": json.dumps(aggregate.attachments),
                    "version": aggregate.version,
                    "edited_at": events[-1].occurred_at if aggregate.version > 1 else None,
                },
            )
        return events
