"""Unit-Tests für den Shared Kernel — ohne DB, laufen immer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest

from backend.domain.shared.command import Command
from backend.domain.shared.event_bus import InProcessEventBus
from backend.domain.shared.event_envelope import EventEnvelope
from tests.backend.doubles import DummyAggregate


def test_event_envelope_rejects_invalid_source():
    with pytest.raises(ValueError):
        EventEnvelope(
            aggregate_type="X",
            aggregate_id=uuid.uuid4(),
            version=1,
            event_type="Whatever",
            payload={},
            source="not-a-real-source",
            correlation_id=uuid.uuid4(),
        )


def test_event_envelope_rejects_version_below_one():
    with pytest.raises(ValueError):
        EventEnvelope(
            aggregate_type="X",
            aggregate_id=uuid.uuid4(),
            version=0,
            event_type="Whatever",
            payload={},
            source="system",
            correlation_id=uuid.uuid4(),
        )


def test_aggregate_root_raise_event_applies_and_bumps_version():
    correlation_id = uuid.uuid4()
    aggregate = DummyAggregate(uuid.uuid4())

    aggregate.increment(3, correlation_id=correlation_id)
    aggregate.increment(4, correlation_id=correlation_id)

    assert aggregate.counter == 7
    assert aggregate.version == 2

    events = aggregate.pull_uncommitted_events()
    assert [e.version for e in events] == [1, 2]
    assert aggregate.pull_uncommitted_events() == []  # zweites Ziehen ist leer


def test_replay_from_events_reproduces_identical_state():
    correlation_id = uuid.uuid4()
    original = DummyAggregate(uuid.uuid4())
    original.increment(3, correlation_id=correlation_id)
    original.increment(4, correlation_id=correlation_id)
    events = original.pull_uncommitted_events()

    rebuilt = DummyAggregate(original.id)
    for event in events:
        rebuilt.apply(event)
        rebuilt.version = event.version

    assert rebuilt.counter == original.counter == 7
    assert rebuilt.version == original.version == 2


def test_in_process_event_bus_dispatches_to_matching_subscribers():
    received: list[EventEnvelope] = []
    bus = InProcessEventBus()
    bus.subscribe("DummyIncremented", received.append)

    aggregate = DummyAggregate(uuid.uuid4())
    aggregate.increment(5, correlation_id=uuid.uuid4())
    bus.publish(aggregate.pull_uncommitted_events())

    assert len(received) == 1
    assert received[0].payload == {"amount": 5}


def test_in_process_event_bus_ignores_unrelated_event_types():
    received: list[EventEnvelope] = []
    bus = InProcessEventBus()
    bus.subscribe("SomethingElseHappened", received.append)

    aggregate = DummyAggregate(uuid.uuid4())
    aggregate.increment(5, correlation_id=uuid.uuid4())
    bus.publish(aggregate.pull_uncommitted_events())

    assert received == []


def test_command_subclass_can_add_required_fields_via_kw_only():
    @dataclass(frozen=True, kw_only=True)
    class DoSomethingCommand(Command):
        target_id: uuid.UUID

    cmd = DoSomethingCommand(target_id=uuid.uuid4(), source="mobile")
    assert cmd.command_id is not None
    assert cmd.source == "mobile"


def test_command_rejects_invalid_source():
    with pytest.raises(ValueError):
        Command(source="carrier-pigeon")
