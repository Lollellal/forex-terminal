"""Basisklasse für Aggregate-Repositories. Siehe BACKEND_ARCHITECTURE.md §2.2.

Der Event-Store-Teil (Laden/Speichern des Event-Stroms eines Aggregates) ist
hier vollständig und wiederverwendbar. Konkrete Repositories (ab Account/
Empire in Implementierungsschritt 2) erben davon und ergänzen ausschließlich
die Synchronisation mit ihrer Aggregate-State-Tabelle (Hybrid-Ansatz aus
BACKEND_ARCHITECTURE.md §2.1) — dieser Schritt baut noch keine konkrete
Aggregate-State-Tabelle, nur das wiederverwendbare Fundament dafür.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sqlalchemy import Connection

from .aggregate_root import AggregateRoot
from .event_envelope import EventEnvelope
from .event_store_protocol import EventStoreProtocol
from .exceptions import EventStreamGapError

TAggregate = TypeVar("TAggregate", bound=AggregateRoot)


class EventSourcedRepository(ABC, Generic[TAggregate]):
    def __init__(self, event_store: EventStoreProtocol) -> None:
        self._event_store = event_store

    @property
    @abstractmethod
    def aggregate_type(self) -> str: ...

    @abstractmethod
    def _blank_instance(self, aggregate_id: uuid.UUID) -> TAggregate:
        """Erzeugt eine leere Aggregate-Instanz für Replay. Darf außer der
        aggregate_id keine weiteren Pflichtdaten voraussetzen — der Zustand
        wird ausschließlich durch das Anwenden der geladenen Events aufgebaut."""

    def load(self, conn: Connection, aggregate_id: uuid.UUID) -> TAggregate:
        events = self._event_store.load_stream(conn, self.aggregate_type, aggregate_id)
        if not events:
            raise LookupError(f"{self.aggregate_type} {aggregate_id} nicht gefunden")
        aggregate = self._blank_instance(aggregate_id)
        expected_version = 0
        for event in events:
            if event.version != expected_version + 1:
                raise EventStreamGapError(
                    f"{self.aggregate_type} {aggregate_id}: erwartete Version "
                    f"{expected_version + 1}, bekam {event.version}"
                )
            aggregate.apply(event)
            expected_version = event.version
        aggregate.version = expected_version
        return aggregate

    def save(self, conn: Connection, aggregate: TAggregate) -> list[EventEnvelope]:
        events = aggregate.pull_uncommitted_events()
        if events:
            self._event_store.append(conn, events)
        return events
