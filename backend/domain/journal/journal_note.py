"""JournalNote-Aggregate. Siehe BACKEND_ARCHITECTURE.md §2.1/§2.2.

Minimal per Definition: nur Text + genau eine Referenz (Allocation ODER
Signal). Trägt bewusst KEINE Trade-Fakten (Ergebnis/R-Multiple/Preise/
Status) — das ist strukturell erzwungen, weil das Aggregate schlicht keine
Felder dafür hat, nicht nur Konvention.
"""

from __future__ import annotations

import uuid

from backend.domain.shared.aggregate_root import AggregateRoot
from backend.domain.shared.event_envelope import EventEnvelope

from . import events


class JournalNote(AggregateRoot):
    aggregate_type = "JournalNote"

    def __init__(self, aggregate_id: uuid.UUID, version: int = 0) -> None:
        super().__init__(aggregate_id, version)
        self.related_allocation_id: uuid.UUID | None = None
        self.related_signal_id: uuid.UUID | None = None
        self.text: str | None = None
        self.attachments: list[str] = []

    @classmethod
    def add(
        cls,
        aggregate_id: uuid.UUID,
        *,
        text: str,
        related_allocation_id: uuid.UUID | None = None,
        related_signal_id: uuid.UUID | None = None,
        attachments: list[str] | None = None,
        source: str,
        correlation_id: uuid.UUID,
    ) -> "JournalNote":
        has_allocation = related_allocation_id is not None
        has_signal = related_signal_id is not None
        if has_allocation == has_signal:
            raise ValueError(
                "JournalNote braucht genau eine Referenz: entweder related_allocation_id "
                "oder related_signal_id, nicht beide oder keine"
            )
        if not text:
            raise ValueError("JournalNote-Text darf nicht leer sein")

        note = cls(aggregate_id)
        note.raise_event(
            events.JOURNAL_NOTE_ADDED,
            {
                "related_allocation_id": str(related_allocation_id) if related_allocation_id else None,
                "related_signal_id": str(related_signal_id) if related_signal_id else None,
                "text": text,
                "attachments": list(attachments or []),
            },
            source=source,
            correlation_id=correlation_id,
        )
        return note

    def edit(
        self,
        *,
        text: str,
        attachments: list[str] | None = None,
        source: str,
        correlation_id: uuid.UUID,
    ) -> None:
        if not text:
            raise ValueError("JournalNote-Text darf nicht leer sein")
        self.raise_event(
            events.JOURNAL_NOTE_EDITED,
            {"text": text, "attachments": list(attachments if attachments is not None else self.attachments)},
            source=source,
            correlation_id=correlation_id,
        )

    def apply(self, event: EventEnvelope) -> None:
        if event.event_type == events.JOURNAL_NOTE_ADDED:
            allocation_id = event.payload["related_allocation_id"]
            signal_id = event.payload["related_signal_id"]
            self.related_allocation_id = uuid.UUID(allocation_id) if allocation_id else None
            self.related_signal_id = uuid.UUID(signal_id) if signal_id else None
            self.text = event.payload["text"]
            self.attachments = list(event.payload["attachments"])
        elif event.event_type == events.JOURNAL_NOTE_EDITED:
            self.text = event.payload["text"]
            self.attachments = list(event.payload["attachments"])
        else:
            raise ValueError(f"JournalNote kennt event_type {event.event_type!r} nicht")
