"""Transaktionale Klammer für die Command-Pipeline.

Siehe BACKEND_ARCHITECTURE.md §2.1 (Hybrid-Ansatz: Event-Append + State-Update
in derselben Transaktion) und §2.4 (Command-Pipeline Schritt 8). Ab Schritt 1
gibt es noch keine Aggregate-State-Tabellen — die Klammer schützt bisher nur
den Event-Append, ist aber schon die Stelle, an der künftige State-Updates
in derselben Transaktion ergänzt werden.

Der konkrete EventStore wird injiziert (Dependency Inversion) statt hier
importiert — Domain Layer kennt nur das EventStoreProtocol, nie die
Infrastructure-Implementierung.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Connection
from sqlalchemy.engine import Engine

from .event_store_protocol import EventStoreProtocol


class UnitOfWork:
    def __init__(self, engine: Engine, event_store: EventStoreProtocol) -> None:
        self._engine = engine
        self.event_store = event_store

    @contextmanager
    def begin(self) -> Iterator[Connection]:
        """Commit bei erfolgreichem Verlassen des Blocks, Rollback bei jeder
        Exception — inklusive ConcurrencyConflictError aus dem Event Store."""
        with self._engine.begin() as conn:
            yield conn
