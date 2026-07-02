"""WeeklyReport-Aggregate. Siehe BACKEND_ARCHITECTURE.md §2.1/§2.2.

Registriert nur Metadaten + eine Storage-Referenz zu einem bereits extern
(Desktop) generierten PDF — keine Report-Generierung, keine KI-Logik hier.
Status-Lifecycle in diesem Schritt: GENERATED -> PUBLISHED (DRAFT/ARCHIVED
sind im DB-Schema vorgesehen, aber kein Codepfad erzeugt sie aktuell)."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from backend.domain.shared.aggregate_root import AggregateRoot
from backend.domain.shared.event_envelope import EventEnvelope

from . import events


class WeeklyReport(AggregateRoot):
    aggregate_type = "WeeklyReport"

    def __init__(self, aggregate_id: uuid.UUID, version: int = 0) -> None:
        super().__init__(aggregate_id, version)
        self.user_id: uuid.UUID | None = None
        self.period_start: date | None = None
        self.period_end: date | None = None
        self.status: str | None = None
        self.content_ref: str | None = None
        self.published_at: datetime | None = None
        self.summary: str | None = None

    @classmethod
    def generate(
        cls,
        aggregate_id: uuid.UUID,
        *,
        user_id: uuid.UUID,
        period_start: date,
        period_end: date,
        content_ref: str,
        summary: str | None = None,
        source: str,
        correlation_id: uuid.UUID,
    ) -> "WeeklyReport":
        if period_end < period_start:
            raise ValueError("period_end darf nicht vor period_start liegen")
        if not content_ref:
            raise ValueError("content_ref darf nicht leer sein — Report ohne PDF-Referenz ist nicht registrierbar")

        report = cls(aggregate_id)
        report.raise_event(
            events.WEEKLY_REPORT_GENERATED,
            {
                "user_id": str(user_id),
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "content_ref": content_ref,
                "summary": summary,
            },
            source=source,
            correlation_id=correlation_id,
        )
        return report

    def publish(self, *, source: str, correlation_id: uuid.UUID) -> None:
        if self.status != "GENERATED":
            raise ValueError(
                f"WeeklyReport {self.id}: publish() erfordert Status 'GENERATED', aktuell {self.status!r}"
            )
        self.raise_event(
            events.WEEKLY_REPORT_PUBLISHED,
            {"published_at": datetime.now(timezone.utc).isoformat()},
            source=source,
            correlation_id=correlation_id,
        )

    def apply(self, event: EventEnvelope) -> None:
        if event.event_type == events.WEEKLY_REPORT_GENERATED:
            self.user_id = uuid.UUID(event.payload["user_id"])
            self.period_start = date.fromisoformat(event.payload["period_start"])
            self.period_end = date.fromisoformat(event.payload["period_end"])
            self.content_ref = event.payload["content_ref"]
            self.summary = event.payload.get("summary")
            self.status = "GENERATED"
        elif event.event_type == events.WEEKLY_REPORT_PUBLISHED:
            self.published_at = datetime.fromisoformat(event.payload["published_at"])
            self.status = "PUBLISHED"
        else:
            raise ValueError(f"WeeklyReport kennt event_type {event.event_type!r} nicht")
