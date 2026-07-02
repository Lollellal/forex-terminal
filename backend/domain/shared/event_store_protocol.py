"""Schnittstelle, gegen die der Domain Layer programmiert (Dependency
Inversion). Die konkrete Implementierung (backend.infrastructure.event_store)
wird von außen injiziert — der Domain Layer darf sie nie importieren, sonst
entsteht ein Zyklus Domain -> Infrastructure -> Domain.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from sqlalchemy import Connection

from .event_envelope import EventEnvelope


class EventStoreProtocol(Protocol):
    def append(self, conn: Connection, events: list[EventEnvelope]) -> None: ...

    def load_stream(
        self, conn: Connection, aggregate_type: str, aggregate_id: uuid.UUID
    ) -> list[EventEnvelope]: ...

    def load_all(
        self, conn: Connection, after_seq: int = 0, limit: int = 1000
    ) -> list[tuple[int, EventEnvelope]]: ...
