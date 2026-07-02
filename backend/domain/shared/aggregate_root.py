"""Aggregate-Basisklasse. Siehe BACKEND_ARCHITECTURE.md §2.2.

Ein Aggregate ist die einzige Instanz mit Schreibautorität über seinen
eigenen Zustand (DOMAIN_ARCHITECTURE.md §1). Zustand ändert sich
ausschließlich über raise_event() — nie durch direktes Setzen von
Attributen von außen.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from .event_envelope import EventEnvelope


class AggregateRoot(ABC):
    aggregate_type: str = ""  # von Subklassen überschrieben, z.B. "TradeAllocation"

    def __init__(self, aggregate_id: uuid.UUID, version: int = 0) -> None:
        if not self.aggregate_type:
            raise NotImplementedError(f"{type(self).__name__} muss aggregate_type setzen")
        self.id = aggregate_id
        self.version = version
        self._uncommitted_events: list[EventEnvelope] = []

    def raise_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        source: str,
        correlation_id: uuid.UUID,
        causation_id: uuid.UUID | None = None,
        device_id: uuid.UUID | None = None,
    ) -> EventEnvelope:
        """Erzeugt ein neues Event, wendet es sofort auf den In-Memory-Zustand an
        und merkt es zum Persistieren vor. Die Versionsnummer wird zentral hier
        vergeben, damit Aggregate-Zustand und Event Store nie auseinanderlaufen."""
        event = EventEnvelope(
            aggregate_type=self.aggregate_type,
            aggregate_id=self.id,
            version=self.version + 1,
            event_type=event_type,
            payload=payload,
            source=source,
            correlation_id=correlation_id,
            causation_id=causation_id,
            device_id=device_id,
        )
        self.apply(event)
        self.version = event.version
        self._uncommitted_events.append(event)
        return event

    @abstractmethod
    def apply(self, event: EventEnvelope) -> None:
        """Mutiert den In-Memory-Zustand anhand eines Events. Läuft sowohl bei
        raise_event() (neues Event) als auch beim Replay aus dem Event Store —
        muss daher deterministisch und frei von Seiteneffekten sein (kein I/O,
        keine Zufallswerte, keine 'jetzt'-Zeitstempel innerhalb von apply())."""

    def pull_uncommitted_events(self) -> list[EventEnvelope]:
        """Gibt alle seit dem letzten Aufruf erzeugten Events zurück und leert
        die interne Liste — wird vom Repository beim Speichern aufgerufen."""
        events = list(self._uncommitted_events)
        self._uncommitted_events.clear()
        return events
