"""Fixtures für die Backend-Integrationstests gegen Supabase/PostgreSQL.

Läuft DATABASE_URL nicht in .env oder ist core.event_store nicht erreichbar
(Migration noch nicht ausgeführt), werden die betroffenen Tests übersprungen
statt rot zu laufen — Unit-Tests (test_shared_kernel.py) sind davon unabhängig.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from backend.infrastructure.db import get_database_url


@pytest.fixture(scope="session")
def db_engine():
    try:
        url = get_database_url()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    from sqlalchemy import create_engine

    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM core.event_store LIMIT 1"))
    except Exception as exc:  # noqa: BLE001 - Skip-Grund soll die echte Ursache zeigen
        pytest.skip(
            f"core.event_store nicht erreichbar — Migration ausgeführt? "
            f"DATABASE_URL korrekt? Ursache: {exc}"
        )
    yield engine
    engine.dispose()
