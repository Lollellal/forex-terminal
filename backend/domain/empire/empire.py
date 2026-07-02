"""Empire-Aggregate. Siehe BACKEND_ARCHITECTURE.md §2.1/§2.2.

Minimal für Implementierungsschritt 2: nur Name/Ownership. Die Account-
Zuordnung lebt bewusst auf dem Account-Aggregate (accounts.empire_id) statt
als Liste hier — sonst zwei Wahrheiten über dieselbe Beziehung. Stage-
Advance/Payout-Historie (BACKEND_ARCHITECTURE.md §2.2) folgen mit dem
Risk-Gate-Schritt.
"""

from __future__ import annotations

import uuid

from backend.domain.shared.aggregate_root import AggregateRoot
from backend.domain.shared.event_envelope import EventEnvelope

from . import events


class Empire(AggregateRoot):
    aggregate_type = "Empire"

    def __init__(self, aggregate_id: uuid.UUID, version: int = 0) -> None:
        super().__init__(aggregate_id, version)
        self.user_id: uuid.UUID | None = None
        self.name: str | None = None

    @classmethod
    def create(
        cls,
        aggregate_id: uuid.UUID,
        *,
        user_id: uuid.UUID,
        name: str,
        source: str,
        correlation_id: uuid.UUID,
    ) -> "Empire":
        if not name:
            raise ValueError("Empire-Name darf nicht leer sein")
        empire = cls(aggregate_id)
        empire.raise_event(
            events.EMPIRE_CREATED,
            {"user_id": str(user_id), "name": name},
            source=source,
            correlation_id=correlation_id,
        )
        return empire

    def apply(self, event: EventEnvelope) -> None:
        if event.event_type == events.EMPIRE_CREATED:
            self.user_id = uuid.UUID(event.payload["user_id"])
            self.name = event.payload["name"]
        else:
            raise ValueError(f"Empire kennt event_type {event.event_type!r} nicht")
