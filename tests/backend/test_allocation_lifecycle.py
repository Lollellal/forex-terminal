"""Integrationstests gegen echtes PostgreSQL (Supabase) — Implementierungs-
schritt 3: TradeAllocation Lifecycle.

Definition of Done: Create -> Confirm -> Open -> Close funktioniert über
Commands + Lifecycle-Service, Status-Übergänge sind erzwungen, die
Projektion für aktive/geschlossene Allocations spiegelt jeden Schritt.
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
from backend.domain.allocation.trade_allocation import TradeAllocation
from backend.domain.risk.policies.consecutive_losses_policy import ConsecutiveLossesPolicy
from backend.domain.risk.policies.same_pair_policy import SamePairOpenPolicy
from backend.domain.risk.risk_gate_service import RiskGateService
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner


def _risk_gate() -> RiskGateService:
    return RiskGateService([SamePairOpenPolicy(), ConsecutiveLossesPolicy()])


@pytest.fixture
def account_ctx(db_engine):
    user_id = uuid.uuid4()
    account_id = uuid.uuid4()
    allocation_ids: list[uuid.UUID] = []

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

    yield user_id, account_id, allocation_ids

    with db_engine.begin() as conn:
        for aid in allocation_ids:
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


def test_create_allocation_persists_and_reloads(db_engine, account_ctx):
    _, account_id, allocation_ids = account_ctx
    repo = AllocationRepository(EventStore())
    allocation = TradeAllocation.create(
        uuid.uuid4(),
        account_id=account_id,
        pair="EURUSD",
        direction="LONG",
        planned_risk_pct=Decimal("1.0"),
        entry_price_planned=Decimal("1.08500"),
        sl_price=Decimal("1.08200"),
        tp_price=Decimal("1.09000"),
        source="system",
        correlation_id=uuid.uuid4(),
    )
    allocation_ids.append(allocation.id)

    with db_engine.begin() as conn:
        repo.save(conn, allocation)

    with db_engine.connect() as conn:
        reloaded = repo.load(conn, allocation.id)
        state_row = conn.execute(
            text("SELECT status, version FROM core.trade_allocations WHERE id = :id"),
            {"id": str(allocation.id)},
        ).one()

    assert reloaded.status == "CREATED"
    assert reloaded.pair == "EURUSD"
    assert reloaded.version == 1
    assert state_row.status == "CREATED"
    assert state_row.version == 1


def test_invalid_transition_is_rejected(db_engine, account_ctx):
    _, account_id, allocation_ids = account_ctx
    allocation = TradeAllocation.create(
        uuid.uuid4(),
        account_id=account_id,
        pair="EURUSD",
        direction="LONG",
        planned_risk_pct=Decimal("1.0"),
        source="system",
        correlation_id=uuid.uuid4(),
    )
    allocation_ids.append(allocation.id)

    with pytest.raises(ValueError):
        allocation.mark_opened(
            applied_risk_pct=Decimal("1.0"),
            source="system",
            correlation_id=uuid.uuid4(),
        )

    with pytest.raises(ValueError):
        allocation.close(
            close_reason="TP", realized_r=Decimal("1.5"), source="system", correlation_id=uuid.uuid4()
        )


def test_full_lifecycle_via_commands_and_service(db_engine, account_ctx):
    _, account_id, allocation_ids = account_ctx
    service = AllocationLifecycleService(AllocationRepository(EventStore()), _risk_gate())
    runner = ProjectionRunner(EventStore())

    with db_engine.begin() as conn:
        allocation = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id,
                pair="GBPUSD",
                direction="SHORT",
                planned_risk_pct=Decimal("1.0"),
                entry_price_planned=Decimal("1.27000"),
            ),
        )
        allocation_ids.append(allocation.id)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT status FROM projections.allocation_overview WHERE allocation_id = :id"),
            {"id": str(allocation.id)},
        ).one()
    assert row.status == "CREATED"

    with db_engine.begin() as conn:
        service.confirm(conn, ConfirmAllocationCommand(allocation_id=allocation.id))
        runner.catch_up(conn)

    with db_engine.begin() as conn:
        service.mark_opened(
            conn,
            MarkAllocationOpenedCommand(allocation_id=allocation.id),
        )
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, applied_risk_pct FROM projections.allocation_overview "
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
                allocation_id=allocation.id, close_reason="TP", realized_r=Decimal("1.8")
            ),
        )
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, close_reason, realized_r FROM projections.allocation_overview "
                "WHERE allocation_id = :id"
            ),
            {"id": str(allocation.id)},
        ).one()
        active_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM projections.allocation_overview "
                "WHERE allocation_id = :id AND status <> 'CLOSED'"
            ),
            {"id": str(allocation.id)},
        ).scalar_one()

    assert row.status == "CLOSED"
    assert row.close_reason == "TP"
    assert row.realized_r == Decimal("1.800")
    assert active_count == 0


def test_signal_snapshot_persists_and_stays_immutable(db_engine, account_ctx):
    _, account_id, allocation_ids = account_ctx
    service = AllocationLifecycleService(AllocationRepository(EventStore()), _risk_gate())
    runner = ProjectionRunner(EventStore())
    snapshot = {
        "ml_score": -0.364,
        "quality": "VALID",
        "edge": "Weak Edge",
        "alignment": "CONTRARY",
        "combo_key": "VALID|Weak Edge|Short|CONTRARY",
        "regime": "RISK_ON",
        "overall_score": None,
        "seasonality_score": None,
        "primary_drivers": None,
        "weekly_report_id": None,
        "report_week": None,
    }

    with db_engine.begin() as conn:
        allocation = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id,
                pair="AUDCAD",
                direction="SHORT",
                planned_risk_pct=Decimal("1.0"),
                signal_snapshot=snapshot,
            ),
        )
        allocation_ids.append(allocation.id)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        core_row = conn.execute(
            text("SELECT signal_snapshot FROM core.trade_allocations WHERE id = :id"),
            {"id": str(allocation.id)},
        ).one()
        proj_row = conn.execute(
            text("SELECT signal_snapshot FROM projections.allocation_overview WHERE allocation_id = :id"),
            {"id": str(allocation.id)},
        ).one()
    assert core_row.signal_snapshot == snapshot
    assert proj_row.signal_snapshot == snapshot

    with db_engine.begin() as conn:
        service.confirm(conn, ConfirmAllocationCommand(allocation_id=allocation.id))
        service.mark_opened(conn, MarkAllocationOpenedCommand(allocation_id=allocation.id))
        service.close(
            conn,
            CloseAllocationCommand(
                allocation_id=allocation.id, close_reason="TP", realized_r=Decimal("1.2")
            ),
        )
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        core_row = conn.execute(
            text("SELECT signal_snapshot, status FROM core.trade_allocations WHERE id = :id"),
            {"id": str(allocation.id)},
        ).one()
    assert core_row.status == "CLOSED"
    assert core_row.signal_snapshot == snapshot  # unveraendert durch confirm/open/close


def test_signal_snapshot_defaults_to_none(db_engine, account_ctx):
    _, account_id, allocation_ids = account_ctx
    repo = AllocationRepository(EventStore())
    allocation = TradeAllocation.create(
        uuid.uuid4(),
        account_id=account_id,
        pair="EURUSD",
        direction="LONG",
        planned_risk_pct=Decimal("1.0"),
        source="system",
        correlation_id=uuid.uuid4(),
    )
    allocation_ids.append(allocation.id)

    with db_engine.begin() as conn:
        repo.save(conn, allocation)

    with db_engine.connect() as conn:
        reloaded = repo.load(conn, allocation.id)

    assert reloaded.signal_snapshot is None
