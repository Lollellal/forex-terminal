"""Event-Contract, verbindlich für jedes Event im System.

Siehe DOMAIN_ARCHITECTURE.md §2.1 und BACKEND_ARCHITECTURE.md §2.1/§2.3.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

VALID_SOURCES = frozenset({"desktop", "mobile", "system", "scheduled-job"})


@dataclass(frozen=True)
class EventEnvelope:
    """Unveränderlicher Fakt. Kein Event ohne eindeutigen aggregate_type +
    aggregate_id (DOMAIN_ARCHITECTURE.md §2.1) — deshalb sind beide Pflichtfelder
    ohne Default, alles andere ist bewusst so restriktiv wie das Envelope selbst."""

    aggregate_type: str
    aggregate_id: uuid.UUID
    version: int
    event_type: str
    payload: dict[str, Any]
    source: str
    correlation_id: uuid.UUID
    causation_id: uuid.UUID | None = None
    device_id: uuid.UUID | None = None
    schema_version: int = 1
    event_id: uuid.UUID = field(default_factory=uuid.uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(f"invalid event source: {self.source!r}")
        if self.version < 1:
            raise ValueError("event version must start at 1")
        if not self.aggregate_type:
            raise ValueError("aggregate_type darf nicht leer sein")
        if not self.event_type:
            raise ValueError("event_type darf nicht leer sein")
