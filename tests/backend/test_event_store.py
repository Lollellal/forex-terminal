"""Integrationstests gegen echtes PostgreSQL (Supabase) — Kernauftrag von
Implementierungsschritt 1: 'Event speichern + wieder laden'.

Setup vor dem ersten Lauf:
    1. DATABASE_URL in .env eintragen (Supabase Connection-URI)
    2. cd backend && alembic upgrade head
    3. pytest tests/backend -v
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from backend.domain.shared.event_envelope import EventEnvelope
from backend.domain.shared.exceptions import ConcurrencyConflictError
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.idempotency import CommandDeduplicator
from tests.backend.doubles import DummyAggregate, DummyRepository


@pytest.fixture
def aggregate_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture(autouse=True)
def _cleanup(db_engine, aggregate_id):
    yield
    with db_engine.begin() as conn:
        conn.execute(
            text("DELETE FROM core.event_store WHERE aggregate_id = :aid"),
            {"aid": str(aggregate_id)},
        )
        conn.execute(
            text("DELETE FROM core.processed_commands WHERE aggregate_id = :aid"),
            {"aid": str(aggregate_id)},
        )


def _make_event(aggregate_id, version, event_type="TestEventHappened", **overrides):
    defaults = dict(
        aggregate_type="TestAggregate",
        aggregate_id=aggregate_id,
        version=version,
        event_type=event_type,
        payload={"note": f"version {version}"},
        source="system",
        correlation_id=uuid.uuid4(),
    )
    defaults.update(overrides)
    return EventEnvelope(**defaults)


def test_append_and_load_stream_roundtrip(db_engine, aggregate_id):
    store = EventStore()
    events = [
        _make_event(aggregate_id, 1),
        _make_event(aggregate_id, 2, event_type="TestEventUpdated"),
    ]

    with db_engine.begin() as conn:
        store.append(conn, events)

    with db_engine.connect() as conn:
        loaded = store.load_stream(conn, "TestAggregate", aggregate_id)

    assert [e.version for e in loaded] == [1, 2]
    assert [e.event_type for e in loaded] == ["TestEventHappened", "TestEventUpdated"]
    assert loaded[0].payload == {"note": "version 1"}
    assert loaded[0].aggregate_id == aggregate_id
    assert loaded[0].source == "system"


def test_duplicate_version_is_rejected_as_concurrency_conflict(db_engine, aggregate_id):
    store = EventStore()
    with db_engine.begin() as conn:
        store.append(conn, [_make_event(aggregate_id, 1)])

    duplicate = _make_event(aggregate_id, 1, event_type="TestEventHappenedAgain")
    with pytest.raises(ConcurrencyConflictError):
        with db_engine.begin() as conn:
            store.append(conn, [duplicate])

    # der abgelehnte Insert darf keine Spur hinterlassen
    with db_engine.connect() as conn:
        loaded = store.load_stream(conn, "TestAggregate", aggregate_id)
    assert len(loaded) == 1
    assert loaded[0].event_type == "TestEventHappened"


def test_repository_save_then_load_roundtrip(db_engine, aggregate_id):
    repo = DummyRepository(EventStore())

    aggregate = DummyAggregate(aggregate_id)
    aggregate.increment(3, correlation_id=uuid.uuid4())
    aggregate.increment(4, correlation_id=uuid.uuid4())

    with db_engine.begin() as conn:
        repo.save(conn, aggregate)

    with db_engine.connect() as conn:
        reloaded = repo.load(conn, aggregate_id)

    assert reloaded.counter == 7
    assert reloaded.version == 2
    assert reloaded.id == aggregate_id


def test_command_deduplicator_prevents_double_processing(db_engine, aggregate_id):
    dedup = CommandDeduplicator()
    command_id = uuid.uuid4()

    with db_engine.begin() as conn:
        assert dedup.already_processed(conn, command_id) is None
        dedup.record(conn, command_id, aggregate_id, {"status": "ok"})

    with db_engine.connect() as conn:
        result = dedup.already_processed(conn, command_id)
    assert result == {"status": "ok"}

    with db_engine.begin() as conn:
        conn.execute(
            text("DELETE FROM core.processed_commands WHERE command_id = :cid"),
            {"cid": str(command_id)},
        )
