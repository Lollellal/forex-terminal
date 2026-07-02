"""Integrationstests gegen echtes PostgreSQL (Supabase) — Implementierungs-
schritt 5: Journal Projection.

Definition of Done: Trade erscheint im Journal nach AllocationCreated,
Status aktualisiert sich nach Open/Close, realized_r/applied_risk_pct werden
korrekt angezeigt, eine Note verändert keine Trade-Fakten, Empire/Account
bleiben von JournalNotes unberührt.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import text

from backend.domain.account.account import Account
from backend.domain.account.account_repository import AccountRepository
from backend.domain.allocation.allocation_lifecycle_service import AllocationLifecycleService
from backend.domain.allocation.allocation_repository import AllocationRepository
from backend.domain.allocation.commands import (
    CloseAllocationCommand,
    ConfirmAllocationCommand,
    CreateAllocationCommand,
    MarkAllocationOpenedCommand,
)
from backend.domain.journal.commands import AddJournalNoteCommand, EditJournalNoteCommand
from backend.domain.journal.journal_note_repository import JournalNoteRepository
from backend.domain.journal.journal_note_service import JournalNoteService
from backend.domain.risk.policies.consecutive_losses_policy import ConsecutiveLossesPolicy
from backend.domain.risk.policies.same_pair_policy import SamePairOpenPolicy
from backend.domain.risk.risk_gate_service import RiskGateService
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner


def _risk_gate() -> RiskGateService:
    return RiskGateService([SamePairOpenPolicy(), ConsecutiveLossesPolicy()])


@pytest.fixture
def journal_ctx(db_engine):
    user_id = uuid.uuid4()
    account_id = uuid.uuid4()
    allocation_ids: list[uuid.UUID] = []
    note_ids: list[uuid.UUID] = []

    with db_engine.begin() as conn:
        conn.execute(
            text("INSERT INTO core.users (id, label) VALUES (:id, 'pytest')"),
            {"id": str(user_id)},
        )
        account = Account.create(
            account_id,
            user_id=user_id,
            empire_id=None,
            account_type="LIVE",
            initial_balance=Decimal("10000.00"),
            initial_equity=Decimal("10000.00"),
            source="system",
            correlation_id=uuid.uuid4(),
        )
        AccountRepository(EventStore()).save(conn, account)

    yield user_id, account_id, allocation_ids, note_ids

    with db_engine.begin() as conn:
        for nid in note_ids:
            conn.execute(text("DELETE FROM core.journal_notes WHERE id = :id"), {"id": str(nid)})
            conn.execute(
                text(
                    "DELETE FROM core.event_store WHERE aggregate_type = 'JournalNote' "
                    "AND aggregate_id = :id"
                ),
                {"id": str(nid)},
            )
        for aid in allocation_ids:
            conn.execute(
                text("DELETE FROM projections.journal_view WHERE allocation_id = :id"), {"id": str(aid)}
            )
            conn.execute(
                text("DELETE FROM projections.allocation_overview WHERE allocation_id = :id"),
                {"id": str(aid)},
            )
            conn.execute(text("DELETE FROM core.trade_allocations WHERE id = :id"), {"id": str(aid)})
            conn.execute(
                text(
                    "DELETE FROM core.event_store WHERE aggregate_type = 'TradeAllocation' "
                    "AND aggregate_id = :id"
                ),
                {"id": str(aid)},
            )
        conn.execute(
            text("DELETE FROM projections.account_balances WHERE account_id = :id"),
            {"id": str(account_id)},
        )
        conn.execute(text("DELETE FROM core.accounts WHERE id = :id"), {"id": str(account_id)})
        conn.execute(
            text(
                "DELETE FROM core.event_store WHERE aggregate_type = 'Account' AND aggregate_id = :id"
            ),
            {"id": str(account_id)},
        )
        conn.execute(text("DELETE FROM core.users WHERE id = :id"), {"id": str(user_id)})


def test_trade_appears_in_journal_after_allocation_created(db_engine, journal_ctx):
    _, account_id, allocation_ids, _ = journal_ctx
    service = AllocationLifecycleService(AllocationRepository(EventStore()), _risk_gate())
    runner = ProjectionRunner(EventStore())

    with db_engine.begin() as conn:
        allocation = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id, pair="EURUSD", direction="LONG", planned_risk_pct=Decimal("1.0")
            ),
        )
        allocation_ids.append(allocation.id)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, pair, direction, planned_risk_pct, account_snapshot, notes "
                "FROM projections.journal_view WHERE allocation_id = :id"
            ),
            {"id": str(allocation.id)},
        ).one()

    assert row.status == "CREATED"
    assert row.pair == "EURUSD"
    assert row.direction == "LONG"
    assert row.planned_risk_pct == Decimal("1.00")
    assert row.account_snapshot["account_type"] == "LIVE"
    assert row.account_snapshot["balance"] == "10000.00"
    assert row.notes == []


def test_status_updates_after_open_and_close(db_engine, journal_ctx):
    _, account_id, allocation_ids, _ = journal_ctx
    service = AllocationLifecycleService(AllocationRepository(EventStore()), _risk_gate())
    runner = ProjectionRunner(EventStore())

    with db_engine.begin() as conn:
        allocation = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id, pair="GBPUSD", direction="SHORT", planned_risk_pct=Decimal("1.0")
            ),
        )
        allocation_ids.append(allocation.id)
        service.confirm(conn, ConfirmAllocationCommand(allocation_id=allocation.id))
        service.mark_opened(conn, MarkAllocationOpenedCommand(allocation_id=allocation.id))
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, applied_risk_pct FROM projections.journal_view "
                "WHERE allocation_id = :id"
            ),
            {"id": str(allocation.id)},
        ).one()
    assert row.status == "OPEN"
    assert row.applied_risk_pct == Decimal("1.00")

    with db_engine.begin() as conn:
        service.close(
            conn,
            CloseAllocationCommand(
                allocation_id=allocation.id, close_reason="TP", realized_r=Decimal("2.1")
            ),
        )
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, close_reason, realized_r FROM projections.journal_view "
                "WHERE allocation_id = :id"
            ),
            {"id": str(allocation.id)},
        ).one()
    assert row.status == "CLOSED"
    assert row.close_reason == "TP"
    assert row.realized_r == Decimal("2.100")


def test_realized_r_and_applied_risk_reflect_adjusted_decision(db_engine, journal_ctx):
    _, account_id, allocation_ids, _ = journal_ctx
    service = AllocationLifecycleService(AllocationRepository(EventStore()), _risk_gate())
    runner = ProjectionRunner(EventStore())

    with db_engine.begin() as conn:
        first = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id, pair="EURUSD", direction="LONG", planned_risk_pct=Decimal("1.0")
            ),
        )
        allocation_ids.append(first.id)
        service.confirm(conn, ConfirmAllocationCommand(allocation_id=first.id))
        service.mark_opened(conn, MarkAllocationOpenedCommand(allocation_id=first.id))

        second = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id, pair="EURUSD", direction="SHORT", planned_risk_pct=Decimal("1.0")
            ),
        )
        allocation_ids.append(second.id)
        service.confirm(conn, ConfirmAllocationCommand(allocation_id=second.id))
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT applied_risk_pct FROM projections.journal_view WHERE allocation_id = :id"
            ),
            {"id": str(second.id)},
        ).one()
    assert row.applied_risk_pct == Decimal("0.50")


def test_adding_note_does_not_change_trade_facts(db_engine, journal_ctx):
    _, account_id, allocation_ids, note_ids = journal_ctx
    service = AllocationLifecycleService(AllocationRepository(EventStore()), _risk_gate())
    note_service = JournalNoteService(JournalNoteRepository(EventStore()))
    runner = ProjectionRunner(EventStore())

    with db_engine.begin() as conn:
        allocation = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id, pair="EURUSD", direction="LONG", planned_risk_pct=Decimal("1.0")
            ),
        )
        allocation_ids.append(allocation.id)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        before = conn.execute(
            text(
                "SELECT status, pair, direction, planned_risk_pct, applied_risk_pct, "
                "closed_at, close_reason, realized_r "
                "FROM projections.journal_view WHERE allocation_id = :id"
            ),
            {"id": str(allocation.id)},
        ).one()

    with db_engine.begin() as conn:
        note = note_service.add(
            conn,
            AddJournalNoteCommand(
                related_allocation_id=allocation.id,
                text="Guter Einstieg, sauberer Breakout",
                attachments=["https://example.com/screenshot.png"],
            ),
        )
        note_ids.append(note.id)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        after = conn.execute(
            text(
                "SELECT status, pair, direction, planned_risk_pct, applied_risk_pct, "
                "closed_at, close_reason, realized_r, notes "
                "FROM projections.journal_view WHERE allocation_id = :id"
            ),
            {"id": str(allocation.id)},
        ).one()

    assert after.status == before.status
    assert after.pair == before.pair
    assert after.direction == before.direction
    assert after.planned_risk_pct == before.planned_risk_pct
    assert after.applied_risk_pct == before.applied_risk_pct
    assert after.closed_at == before.closed_at
    assert after.close_reason == before.close_reason
    assert after.realized_r == before.realized_r
    assert len(after.notes) == 1
    assert after.notes[0]["text"] == "Guter Einstieg, sauberer Breakout"
    assert after.notes[0]["attachments"] == ["https://example.com/screenshot.png"]

    with db_engine.begin() as conn:
        note_service.edit(
            conn, EditJournalNoteCommand(note_id=note.id, text="Nachtrag: TP zu früh gesetzt")
        )
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        after_edit = conn.execute(
            text("SELECT notes FROM projections.journal_view WHERE allocation_id = :id"),
            {"id": str(allocation.id)},
        ).one()
    assert len(after_edit.notes) == 1
    assert after_edit.notes[0]["text"] == "Nachtrag: TP zu früh gesetzt"


def test_journal_note_leaves_account_and_empire_untouched(db_engine, journal_ctx):
    _, account_id, allocation_ids, note_ids = journal_ctx
    service = AllocationLifecycleService(AllocationRepository(EventStore()), _risk_gate())
    note_service = JournalNoteService(JournalNoteRepository(EventStore()))
    runner = ProjectionRunner(EventStore())

    with db_engine.begin() as conn:
        allocation = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id, pair="EURUSD", direction="LONG", planned_risk_pct=Decimal("1.0")
            ),
        )
        allocation_ids.append(allocation.id)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        account_before = conn.execute(
            text("SELECT balance, equity, version FROM core.accounts WHERE id = :id"),
            {"id": str(account_id)},
        ).one()
        projection_before = conn.execute(
            text("SELECT balance, equity FROM projections.account_balances WHERE account_id = :id"),
            {"id": str(account_id)},
        ).one()
        empire_count_before = conn.execute(text("SELECT COUNT(*) FROM core.empires")).scalar_one()

    with db_engine.begin() as conn:
        note = note_service.add(
            conn,
            AddJournalNoteCommand(related_allocation_id=allocation.id, text="Nur ein Kommentar"),
        )
        note_ids.append(note.id)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        account_after = conn.execute(
            text("SELECT balance, equity, version FROM core.accounts WHERE id = :id"),
            {"id": str(account_id)},
        ).one()
        projection_after = conn.execute(
            text("SELECT balance, equity FROM projections.account_balances WHERE account_id = :id"),
            {"id": str(account_id)},
        ).one()
        empire_count_after = conn.execute(text("SELECT COUNT(*) FROM core.empires")).scalar_one()

    assert account_after == account_before
    assert projection_after == projection_before
    assert empire_count_after == empire_count_before
