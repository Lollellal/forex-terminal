"""Idempotenz-Grundlage für die Command-Pipeline (Schritt 3 in
BACKEND_ARCHITECTURE.md §2.4). Verhindert doppelte Verarbeitung, wenn ein
Client (v.a. Mobile bei instabiler Verbindung) denselben Command erneut
sendet, weil er keine Antwort auf den ersten Versuch erhalten hat.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import Connection, text

_CHECK_SQL = text(
    "SELECT result_summary FROM core.processed_commands WHERE command_id = :command_id"
)

_INSERT_SQL = text(
    """
    INSERT INTO core.processed_commands (command_id, aggregate_id, result_summary)
    VALUES (:command_id, :aggregate_id, CAST(:result_summary AS jsonb))
    """
)


class CommandDeduplicator:
    def already_processed(
        self, conn: Connection, command_id: uuid.UUID
    ) -> dict[str, Any] | None:
        row = conn.execute(_CHECK_SQL, {"command_id": str(command_id)}).first()
        if row is None:
            return None
        result = row[0]
        return result if isinstance(result, dict) else json.loads(result)

    def record(
        self,
        conn: Connection,
        command_id: uuid.UUID,
        aggregate_id: uuid.UUID,
        result_summary: dict[str, Any],
    ) -> None:
        conn.execute(
            _INSERT_SQL,
            {
                "command_id": str(command_id),
                "aggregate_id": str(aggregate_id),
                "result_summary": json.dumps(result_summary),
            },
        )
