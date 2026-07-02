"""DB-Connection-Dependencies für den API Layer.

Zwei Varianten pro FastAPI-Konvention (Dependencies mit ``yield``, siehe
BACKEND_ARCHITECTURE.md §2.4/§2.6): ``get_write_conn`` öffnet eine
Transaktion (Event-Append + State-Update + Projection-Catch-up atomar, wie
in allen bisherigen Schritten), ``get_read_conn`` ist reines Lesen ohne
Transaktionsklammer. Wirft ein Endpoint eine Exception, rollt
``engine.begin()`` automatisch zurück, bevor der registrierte Exception-
Handler (siehe app.py) die HTTP-Antwort baut — FastAPI propagiert die
Exception zuerst durch die Dependency, dann erst zum Handler.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from sqlalchemy import Connection
from sqlalchemy.engine import Engine

from backend.infrastructure.db import get_engine


@lru_cache
def _engine() -> Engine:
    return get_engine()


def get_write_conn() -> Iterator[Connection]:
    with _engine().begin() as conn:
        yield conn


def get_read_conn() -> Iterator[Connection]:
    with _engine().connect() as conn:
        yield conn
