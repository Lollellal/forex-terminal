"""Test-Doubles für den Shared Kernel. Kein Produktionscode — es existiert
absichtlich noch kein echtes Aggregate (Implementierungsschritt 1 baut nur
das Fundament), diese Dummy-Klasse beweist aber, dass AggregateRoot und
EventSourcedRepository tatsächlich benutzbar sind.
"""

from __future__ import annotations

import uuid

from backend.domain.shared.aggregate_root import AggregateRoot
from backend.domain.shared.event_envelope import EventEnvelope
from backend.domain.shared.repository import EventSourcedRepository


class DummyAggregate(AggregateRoot):
    aggregate_type = "DummyAggregate"

    def __init__(self, aggregate_id: uuid.UUID) -> None:
        super().__init__(aggregate_id)
        self.counter = 0

    def apply(self, event: EventEnvelope) -> None:
        if event.event_type == "DummyIncremented":
            self.counter += event.payload["amount"]

    def increment(self, amount: int, *, correlation_id: uuid.UUID) -> None:
        self.raise_event(
            "DummyIncremented",
            {"amount": amount},
            source="system",
            correlation_id=correlation_id,
        )


class DummyRepository(EventSourcedRepository[DummyAggregate]):
    aggregate_type = "DummyAggregate"

    def _blank_instance(self, aggregate_id: uuid.UUID) -> DummyAggregate:
        return DummyAggregate(aggregate_id)
