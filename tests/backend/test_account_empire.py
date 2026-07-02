"""Integrationstests gegen echtes PostgreSQL (Supabase) — Implementierungs-
schritt 2: Account + Empire State.

Definition of Done: Account erstellen, Balance aktualisieren und
Empire-Gesamtkapital aus den Projektionen korrekt berechnen.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import text

from backend.domain.account.account import Account
from backend.domain.account.account_repository import AccountRepository
from backend.domain.empire.empire import Empire
from backend.domain.empire.empire_repository import EmpireRepository
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner


@pytest.fixture
def user_id(db_engine):
    uid = uuid.uuid4()
    account_ids: list[uuid.UUID] = []
    empire_ids: list[uuid.UUID] = []

    with db_engine.begin() as conn:
        conn.execute(
            text("INSERT INTO core.users (id, label) VALUES (:id, 'pytest')"),
            {"id": str(uid)},
        )

    yield uid, account_ids, empire_ids

    with db_engine.begin() as conn:
        for aid in account_ids:
            conn.execute(
                text("DELETE FROM projections.account_balances WHERE account_id = :id"),
                {"id": str(aid)},
            )
            conn.execute(text("DELETE FROM core.accounts WHERE id = :id"), {"id": str(aid)})
            conn.execute(
                text(
                    "DELETE FROM core.event_store WHERE aggregate_type = 'Account' "
                    "AND aggregate_id = :id"
                ),
                {"id": str(aid)},
            )
        for eid in empire_ids:
            conn.execute(
                text("DELETE FROM projections.empire_overview WHERE empire_id = :id"),
                {"id": str(eid)},
            )
            conn.execute(text("DELETE FROM core.empires WHERE id = :id"), {"id": str(eid)})
            conn.execute(
                text(
                    "DELETE FROM core.event_store WHERE aggregate_type = 'Empire' "
                    "AND aggregate_id = :id"
                ),
                {"id": str(eid)},
            )
        conn.execute(text("DELETE FROM core.users WHERE id = :id"), {"id": str(uid)})


def test_account_created_persists_and_reloads(db_engine, user_id):
    uid, account_ids, _ = user_id
    repo = AccountRepository(EventStore())
    account_id = uuid.uuid4()
    account_ids.append(account_id)

    account = Account.create(
        account_id,
        user_id=uid,
        empire_id=None,
        account_type="LIVE",
        initial_balance=Decimal("10000.00"),
        initial_equity=Decimal("10000.00"),
        source="system",
        correlation_id=uuid.uuid4(),
    )
    with db_engine.begin() as conn:
        repo.save(conn, account)

    with db_engine.connect() as conn:
        reloaded = repo.load(conn, account_id)
        state_row = conn.execute(
            text("SELECT balance, equity, status, version FROM core.accounts WHERE id = :id"),
            {"id": str(account_id)},
        ).one()

    assert reloaded.balance == Decimal("10000.00")
    assert reloaded.status == "ACTIVE"
    assert reloaded.version == 1
    assert state_row.balance == Decimal("10000.00")
    assert state_row.status == "ACTIVE"
    assert state_row.version == 1


def test_update_balance_appends_event_and_updates_state_table(db_engine, user_id):
    uid, account_ids, _ = user_id
    repo = AccountRepository(EventStore())
    account_id = uuid.uuid4()
    account_ids.append(account_id)

    account = Account.create(
        account_id,
        user_id=uid,
        empire_id=None,
        account_type="LIVE",
        initial_balance=Decimal("10000.00"),
        initial_equity=Decimal("10000.00"),
        source="system",
        correlation_id=uuid.uuid4(),
    )
    with db_engine.begin() as conn:
        repo.save(conn, account)

    with db_engine.begin() as conn:
        account = repo.load(conn, account_id)
        account.update_balance(
            balance=Decimal("10250.50"),
            equity=Decimal("10180.00"),
            source="system",
            correlation_id=uuid.uuid4(),
        )
        repo.save(conn, account)

    with db_engine.connect() as conn:
        reloaded = repo.load(conn, account_id)
        state_row = conn.execute(
            text("SELECT balance, equity, version FROM core.accounts WHERE id = :id"),
            {"id": str(account_id)},
        ).one()

    assert reloaded.balance == Decimal("10250.50")
    assert reloaded.equity == Decimal("10180.00")
    assert reloaded.version == 2
    assert state_row.balance == Decimal("10250.50")
    assert state_row.version == 2


def test_update_balance_on_closed_account_is_rejected(db_engine, user_id):
    uid, account_ids, _ = user_id
    repo = AccountRepository(EventStore())
    account_id = uuid.uuid4()
    account_ids.append(account_id)

    account = Account.create(
        account_id,
        user_id=uid,
        empire_id=None,
        account_type="LIVE",
        initial_balance=Decimal("1000.00"),
        initial_equity=Decimal("1000.00"),
        source="system",
        correlation_id=uuid.uuid4(),
    )
    account.status = "CLOSED"  # in-memory only, kein Event — reicht für den Invarianten-Check

    with pytest.raises(ValueError):
        account.update_balance(
            balance=Decimal("500.00"),
            equity=Decimal("500.00"),
            source="system",
            correlation_id=uuid.uuid4(),
        )


def test_account_balance_projection_reflects_created_and_updated(db_engine, user_id):
    uid, account_ids, _ = user_id
    repo = AccountRepository(EventStore())
    runner = ProjectionRunner(EventStore())
    account_id = uuid.uuid4()
    account_ids.append(account_id)

    account = Account.create(
        account_id,
        user_id=uid,
        empire_id=None,
        account_type="LIVE",
        initial_balance=Decimal("5000.00"),
        initial_equity=Decimal("5000.00"),
        source="system",
        correlation_id=uuid.uuid4(),
    )
    with db_engine.begin() as conn:
        repo.save(conn, account)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT balance, equity, status FROM projections.account_balances "
                "WHERE account_id = :id"
            ),
            {"id": str(account_id)},
        ).one()
    assert row.balance == Decimal("5000.00")
    assert row.status == "ACTIVE"

    with db_engine.begin() as conn:
        account = repo.load(conn, account_id)
        account.update_balance(
            balance=Decimal("5500.00"),
            equity=Decimal("5400.00"),
            source="system",
            correlation_id=uuid.uuid4(),
        )
        repo.save(conn, account)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT balance, equity FROM projections.account_balances WHERE account_id = :id"),
            {"id": str(account_id)},
        ).one()
    assert row.balance == Decimal("5500.00")
    assert row.equity == Decimal("5400.00")


def test_empire_overview_projection_sums_total_capital_across_accounts(db_engine, user_id):
    uid, account_ids, empire_ids = user_id
    account_repo = AccountRepository(EventStore())
    empire_repo = EmpireRepository(EventStore())
    runner = ProjectionRunner(EventStore())

    empire_id = uuid.uuid4()
    empire_ids.append(empire_id)
    empire = Empire.create(
        empire_id,
        user_id=uid,
        name="Test-Empire",
        source="system",
        correlation_id=uuid.uuid4(),
    )

    account_a_id = uuid.uuid4()
    account_b_id = uuid.uuid4()
    account_ids.extend([account_a_id, account_b_id])
    account_a = Account.create(
        account_a_id,
        user_id=uid,
        empire_id=empire_id,
        account_type="LIVE",
        initial_balance=Decimal("10000.00"),
        initial_equity=Decimal("10000.00"),
        source="system",
        correlation_id=uuid.uuid4(),
    )
    account_b = Account.create(
        account_b_id,
        user_id=uid,
        empire_id=empire_id,
        account_type="PROP_FIRM",
        initial_balance=Decimal("25000.00"),
        initial_equity=Decimal("25000.00"),
        source="system",
        correlation_id=uuid.uuid4(),
    )

    with db_engine.begin() as conn:
        empire_repo.save(conn, empire)
        account_repo.save(conn, account_a)
        account_repo.save(conn, account_b)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        overview = conn.execute(
            text(
                "SELECT name, account_count, total_balance, total_equity "
                "FROM projections.empire_overview WHERE empire_id = :id"
            ),
            {"id": str(empire_id)},
        ).one()

    assert overview.name == "Test-Empire"
    assert overview.account_count == 2
    assert overview.total_balance == Decimal("35000.00")
    assert overview.total_equity == Decimal("35000.00")

    with db_engine.begin() as conn:
        account_a = account_repo.load(conn, account_a_id)
        account_a.update_balance(
            balance=Decimal("10500.00"),
            equity=Decimal("10500.00"),
            source="system",
            correlation_id=uuid.uuid4(),
        )
        account_repo.save(conn, account_a)
        runner.catch_up(conn)

    with db_engine.connect() as conn:
        overview = conn.execute(
            text("SELECT total_balance FROM projections.empire_overview WHERE empire_id = :id"),
            {"id": str(empire_id)},
        ).one()
    assert overview.total_balance == Decimal("35500.00")
