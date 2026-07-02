"""Konkrete Persistenz für core.event_store.

Einziger Schreib-/Lesepfad für den Event Store — siehe
BACKEND_ARCHITECTURE.md §2.1/§2.3. Erfüllt strukturell
``backend.domain.shared.event_store_protocol.EventStoreProtocol``
(keine Vererbung nötig, Protocol ist strukturell/duck-typed).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterable

from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from backend.domain.shared.event_envelope import EventEnvelope
from backend.domain.shared.exceptions import ConcurrencyConflictError

_INSERT_SQL = text(
    """
    INSERT INTO core.event_store
        (event_id, aggregate_type, aggregate_id, version, event_type,
         schema_version, payload, source, device_id, correlation_id,
         causation_id, occurred_at)
    VALUES
        (:event_id, :aggregate_type, :aggregate_id, :version, :event_type,
         :schema_version, CAST(:payload AS jsonb), :source, :device_id,
         :correlation_id, :causation_id, :occurred_at)
    """
)

_LOAD_STREAM_SQL = text(
    """
    SELECT event_id, aggregate_type, aggregate_id, version, event_type,
           schema_version, payload, source, device_id, correlation_id,
           causation_id, occurred_at
    FROM core.event_store
    WHERE aggregate_type = :aggregate_type AND aggregate_id = :aggregate_id
    ORDER BY version ASC
    """
)

_LOAD_ALL_SQL = text(
    """
    SELECT global_seq, event_id, aggregate_type, aggregate_id, version,
           event_type, schema_version, payload, source, device_id,
           correlation_id, causation_id, occurred_at
    FROM core.event_store
    WHERE global_seq > :after_seq
    ORDER BY global_seq ASC
    LIMIT :limit
    """
)


class EventStore:
    def append(self, conn: Connection, events: Iterable[EventEnvelope]) -> None:
        """Persistiert alle übergebenen Events. Schlägt eines fehl (z.B. weil
        die Version bereits vergeben ist), bricht die gesamte Operation ab —
        das äußere ``engine.begin()`` rollt die Transaktion zurück. Das ist
        gewollt: Die Events eines einzelnen Commands sind atomar."""
        for event in events:
            try:
                conn.execute(_INSERT_SQL, _event_to_params(event))
            except IntegrityError as exc:
                raise ConcurrencyConflictError(
                    f"{event.aggregate_type} {event.aggregate_id} Version {event.version} "
                    "existiert bereits — konkurrierender Schreibzugriff"
                ) from exc

    def load_stream(
        self, conn: Connection, aggregate_type: str, aggregate_id: uuid.UUID
    ) -> list[EventEnvelope]:
        rows = (
            conn.execute(
                _LOAD_STREAM_SQL,
                {"aggregate_type": aggregate_type, "aggregate_id": str(aggregate_id)},
            )
            .mappings()
            .all()
        )
        return [_row_to_envelope(row) for row in rows]

    def load_all(
        self, conn: Connection, after_seq: int = 0, limit: int = 1000
    ) -> list[tuple[int, EventEnvelope]]:
        """Für Replay/Projection-Updates: alle Events in globaler Reihenfolge
        nach einer Checkpoint-Position (BACKEND_ARCHITECTURE.md §2.3)."""
        rows = conn.execute(_LOAD_ALL_SQL, {"after_seq": after_seq, "limit": limit}).mappings().all()
        return [(row["global_seq"], _row_to_envelope(row)) for row in rows]


def _event_to_params(event: EventEnvelope) -> dict[str, Any]:
    return {
        "event_id": str(event.event_id),
        "aggregate_type": event.aggregate_type,
        "aggregate_id": str(event.aggregate_id),
        "version": event.version,
        "event_type": event.event_type,
        "schema_version": event.schema_version,
        "payload": json.dumps(event.payload),
        "source": event.source,
        "device_id": str(event.device_id) if event.device_id else None,
        "correlation_id": str(event.correlation_id),
        "causation_id": str(event.causation_id) if event.causation_id else None,
        "occurred_at": event.occurred_at,
    }


def _row_to_envelope(row: Any) -> EventEnvelope:
    payload = row["payload"]
    if not isinstance(payload, dict):
        payload = json.loads(payload)
    return EventEnvelope(
        aggregate_type=row["aggregate_type"],
        aggregate_id=uuid.UUID(str(row["aggregate_id"])),
        version=row["version"],
        event_type=row["event_type"],
        payload=payload,
        source=row["source"],
        correlation_id=uuid.UUID(str(row["correlation_id"])),
        causation_id=uuid.UUID(str(row["causation_id"])) if row["causation_id"] else None,
        device_id=uuid.UUID(str(row["device_id"])) if row["device_id"] else None,
        schema_version=row["schema_version"],
        event_id=uuid.UUID(str(row["event_id"])),
        occurred_at=row["occurred_at"],
    )
