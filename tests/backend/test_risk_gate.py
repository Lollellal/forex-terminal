"""Integrationstests gegen echtes PostgreSQL (Supabase) — Implementierungs-
schritt 4: Risk Gate.

Definition of Done: normaler Trade -> ALLOW, Same-Pair-Already-Open -> ADJUST
0.5, zwei Verluste in Folge -> ADJUST 0.5, eine REJECT-Decision verhindert
Confirm/Open tatsächlich (State-Übergang wird gar nicht erst versucht).
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
from backend.domain.risk.decision import REJECT, RiskGateDecision
from backend.domain.risk.exceptions import RiskGateRejectedError
from backend.domain.risk.policies.consecutive_losses_policy import ConsecutiveLossesPolicy
from backend.domain.risk.policies.same_pair_policy import SamePairOpenPolicy
from backend.domain.risk.risk_gate_service import RiskGateService
from backend.infrastructure.event_store import EventStore


class _AlwaysRejectPolicy:
    """Test-Double: keine der beiden echten Policies produziert REJECT —
    dieser Double beweist, dass der Enforcement-Mechanismus selbst
    funktioniert (REJECT blockiert den State-Übergang), unabhängig davon,
    welche fachliche Policy irgendwann REJECT auslöst."""

    policy_key = "always-reject"

    def evaluate(self, context, config):
        return RiskGateDecision(decision_type=REJECT, risk_pct=None, reason="Test-Reject")


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


def _create_and_confirm(conn, service, account_id, pair, allocation_ids, risk=Decimal("1.0")):
    allocation = service.create(
        conn,
        CreateAllocationCommand(account_id=account_id, pair=pair, direction="LONG", planned_risk_pct=risk),
    )
    allocation_ids.append(allocation.id)
    return service.confirm(conn, ConfirmAllocationCommand(allocation_id=allocation.id))


def test_normal_trade_is_allowed(db_engine, account_ctx):
    _, account_id, allocation_ids = account_ctx
    service = AllocationLifecycleService(
        AllocationRepository(EventStore()),
        RiskGateService([SamePairOpenPolicy(), ConsecutiveLossesPolicy()]),
    )

    with db_engine.begin() as conn:
        allocation = _create_and_confirm(conn, service, account_id, "EURUSD", allocation_ids)

    assert allocation.status == "CONFIRMED"
    assert allocation.applied_risk_pct == Decimal("1.0")


def test_same_pair_already_open_adjusts_risk_to_half(db_engine, account_ctx):
    _, account_id, allocation_ids = account_ctx
    service = AllocationLifecycleService(
        AllocationRepository(EventStore()),
        RiskGateService([SamePairOpenPolicy(), ConsecutiveLossesPolicy()]),
    )

    with db_engine.begin() as conn:
        first = _create_and_confirm(conn, service, account_id, "EURUSD", allocation_ids)
        service.mark_opened(conn, MarkAllocationOpenedCommand(allocation_id=first.id))

    with db_engine.begin() as conn:
        second = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id, pair="EURUSD", direction="SHORT", planned_risk_pct=Decimal("1.0")
            ),
        )
        allocation_ids.append(second.id)
        second = service.confirm(conn, ConfirmAllocationCommand(allocation_id=second.id))

    assert second.status == "CONFIRMED"
    assert second.applied_risk_pct == Decimal("0.5")


def test_two_consecutive_losses_adjust_risk_to_half(db_engine, account_ctx):
    _, account_id, allocation_ids = account_ctx
    service = AllocationLifecycleService(
        AllocationRepository(EventStore()),
        RiskGateService([SamePairOpenPolicy(), ConsecutiveLossesPolicy()]),
    )

    with db_engine.begin() as conn:
        for pair in ("EURUSD", "GBPUSD"):
            allocation = _create_and_confirm(conn, service, account_id, pair, allocation_ids)
            service.mark_opened(conn, MarkAllocationOpenedCommand(allocation_id=allocation.id))
            service.close(
                conn,
                CloseAllocationCommand(
                    allocation_id=allocation.id, close_reason="SL", realized_r=Decimal("-1.0")
                ),
            )

    with db_engine.begin() as conn:
        third = service.create(
            conn,
            CreateAllocationCommand(
                account_id=account_id, pair="AUDUSD", direction="LONG", planned_risk_pct=Decimal("1.0")
            ),
        )
        allocation_ids.append(third.id)
        third = service.confirm(conn, ConfirmAllocationCommand(allocation_id=third.id))

    assert third.status == "CONFIRMED"
    assert third.applied_risk_pct == Decimal("0.5")


def test_reject_decision_blocks_confirm_and_open(db_engine, account_ctx):
    """_AlwaysRejectPolicy wird nur ausgewertet, wenn core.risk_policies eine
    passende, aktivierte Config für ihren policy_key hat (Registry-Prinzip,
    siehe RiskGateService.evaluate) — deshalb seedet dieser Test kurz eine
    eigene GLOBAL-Config statt die Registry zu umgehen."""
    _, account_id, allocation_ids = account_ctx
    with db_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO core.risk_policies
                    (id, policy_key, name, description, evaluation_mode, scope_type,
                     scope_id, priority, enabled, adjusted_risk_pct)
                VALUES
                    ('always-reject-test', 'always-reject', 'Test Reject', 'nur für Tests',
                     'SYNC_GATE', 'GLOBAL', NULL, 1, TRUE, NULL)
                """
            )
        )

    try:
        service = AllocationLifecycleService(
            AllocationRepository(EventStore()), RiskGateService([_AlwaysRejectPolicy()])
        )

        with db_engine.begin() as conn:
            allocation = service.create(
                conn,
                CreateAllocationCommand(
                    account_id=account_id, pair="EURUSD", direction="LONG", planned_risk_pct=Decimal("1.0")
                ),
            )
            allocation_ids.append(allocation.id)

            with pytest.raises(RiskGateRejectedError):
                service.confirm(conn, ConfirmAllocationCommand(allocation_id=allocation.id))

        with db_engine.connect() as conn:
            state_row = conn.execute(
                text("SELECT status, version FROM core.trade_allocations WHERE id = :id"),
                {"id": str(allocation.id)},
            ).one()
        assert state_row.status == "CREATED"
        assert state_row.version == 1

        allow_gate = RiskGateService([SamePairOpenPolicy(), ConsecutiveLossesPolicy()])
        allow_service = AllocationLifecycleService(AllocationRepository(EventStore()), allow_gate)
        with db_engine.begin() as conn:
            allow_service.confirm(conn, ConfirmAllocationCommand(allocation_id=allocation.id))

        reject_only_service = AllocationLifecycleService(
            AllocationRepository(EventStore()), RiskGateService([_AlwaysRejectPolicy()])
        )
        with db_engine.begin() as conn:
            with pytest.raises(RiskGateRejectedError):
                reject_only_service.mark_opened(
                    conn, MarkAllocationOpenedCommand(allocation_id=allocation.id)
                )
    finally:
        with db_engine.begin() as conn:
            conn.execute(
                text("DELETE FROM core.risk_policies WHERE id = 'always-reject-test'")
            )

    with db_engine.connect() as conn:
        state_row = conn.execute(
            text("SELECT status, version FROM core.trade_allocations WHERE id = :id"),
            {"id": str(allocation.id)},
        ).one()
    assert state_row.status == "CONFIRMED"
    assert state_row.version == 2
