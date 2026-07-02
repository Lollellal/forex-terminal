"""WeeklyReport-Repository: Event-Append + State-Tabellen-Update in einer
Transaktion (Hybrid-Ansatz, BACKEND_ARCHITECTURE.md §2.1)."""

from __future__ import annotations

import uuid

from sqlalchemy import Connection, text

from backend.domain.shared.event_envelope import EventEnvelope
from backend.domain.shared.event_store_protocol import EventStoreProtocol
from backend.domain.shared.repository import EventSourcedRepository

from .weekly_report import WeeklyReport

_UPSERT_SQL = text(
    """
    INSERT INTO core.weekly_reports
        (id, user_id, period_start, period_end, status, content_ref, summary, published_at, version, updated_at)
    VALUES
        (:id, :user_id, :period_start, :period_end, :status, :content_ref, :summary, :published_at, :version, now())
    ON CONFLICT (id) DO UPDATE SET
        status       = EXCLUDED.status,
        content_ref  = EXCLUDED.content_ref,
        published_at = EXCLUDED.published_at,
        version      = EXCLUDED.version,
        updated_at   = now()
    """
)


class WeeklyReportRepository(EventSourcedRepository[WeeklyReport]):
    def __init__(self, event_store: EventStoreProtocol) -> None:
        super().__init__(event_store)

    @property
    def aggregate_type(self) -> str:
        return "WeeklyReport"

    def _blank_instance(self, aggregate_id: uuid.UUID) -> WeeklyReport:
        return WeeklyReport(aggregate_id)

    def save(self, conn: Connection, aggregate: WeeklyReport) -> list[EventEnvelope]:
        events = super().save(conn, aggregate)
        if events:
            conn.execute(
                _UPSERT_SQL,
                {
                    "id": str(aggregate.id),
                    "user_id": str(aggregate.user_id),
                    "period_start": aggregate.period_start,
                    "period_end": aggregate.period_end,
                    "status": aggregate.status,
                    "content_ref": aggregate.content_ref,
                    "summary": aggregate.summary,
                    "published_at": aggregate.published_at,
                    "version": aggregate.version,
                },
            )
        return events
