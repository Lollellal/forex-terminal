"""Integrationstests gegen echtes PostgreSQL (Supabase) über den FastAPI
TestClient — Implementierungsschritt 7: Mobile MVP Data Contract.

Deckt die drei in MOBILE_DATA_CONTRACT.md identifizierten Lücken ab:
GET /users/{id}/portfolio (Home, 1 Request), user_id/status=ACTIVE-Filter
auf GET /allocations, GET /empires/{id}/accounts.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.api.app import app

client = TestClient(app)


@pytest.fixture
def cleanup_ids(db_engine):
    ids = {"user": [], "empire": [], "account": [], "allocation": [], "note": []}
    yield ids
    with db_engine.begin() as conn:
        for nid in ids["note"]:
            conn.execute(text("DELETE FROM core.journal_notes WHERE id = :id"), {"id": nid})
            conn.execute(
                text(
                    "DELETE FROM core.event_store WHERE aggregate_type = 'JournalNote' AND aggregate_id = :id"
                ),
                {"id": nid},
            )
        for aid in ids["allocation"]:
            conn.execute(text("DELETE FROM projections.journal_view WHERE allocation_id = :id"), {"id": aid})
            conn.execute(
                text("DELETE FROM projections.allocation_overview WHERE allocation_id = :id"), {"id": aid}
            )
            conn.execute(text("DELETE FROM core.trade_allocations WHERE id = :id"), {"id": aid})
            conn.execute(
                text(
                    "DELETE FROM core.event_store WHERE aggregate_type = 'TradeAllocation' AND aggregate_id = :id"
                ),
                {"id": aid},
            )
        for acc_id in ids["account"]:
            conn.execute(text("DELETE FROM projections.account_balances WHERE account_id = :id"), {"id": acc_id})
            conn.execute(text("DELETE FROM core.accounts WHERE id = :id"), {"id": acc_id})
            conn.execute(
                text("DELETE FROM core.event_store WHERE aggregate_type = 'Account' AND aggregate_id = :id"),
                {"id": acc_id},
            )
        for eid in ids["empire"]:
            conn.execute(text("DELETE FROM projections.empire_overview WHERE empire_id = :id"), {"id": eid})
            conn.execute(text("DELETE FROM core.empires WHERE id = :id"), {"id": eid})
            conn.execute(
                text("DELETE FROM core.event_store WHERE aggregate_type = 'Empire' AND aggregate_id = :id"),
                {"id": eid},
            )
        for uid in ids["user"]:
            conn.execute(text("DELETE FROM core.users WHERE id = :id"), {"id": uid})


def _make_user(db_engine, cleanup_ids) -> str:
    user_id = str(uuid.uuid4())
    with db_engine.begin() as conn:
        conn.execute(text("INSERT INTO core.users (id, label) VALUES (:id, 'pytest-mobile')"), {"id": user_id})
    cleanup_ids["user"].append(user_id)
    return user_id


def _create_account(cleanup_ids, user_id, balance="10000.00", empire_id=None, account_type="LIVE") -> str:
    payload = {
        "user_id": user_id,
        "account_type": account_type,
        "initial_balance": balance,
        "initial_equity": balance,
    }
    if empire_id is not None:
        payload["empire_id"] = empire_id
    account_id = client.post("/accounts", json=payload).json()["id"]
    cleanup_ids["account"].append(account_id)
    return account_id


def _create_allocation(cleanup_ids, account_id, pair="EURUSD", direction="LONG") -> str:
    allocation_id = client.post(
        "/allocations",
        json={"account_id": account_id, "pair": pair, "direction": direction, "planned_risk_pct": "1.0"},
    ).json()["id"]
    cleanup_ids["allocation"].append(allocation_id)
    return allocation_id


def test_home_portfolio_single_request_combines_everything(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)

    empire = client.post("/empires", json={"user_id": user_id, "name": "Mobile-Empire"}).json()
    cleanup_ids["empire"].append(empire["id"])
    _create_account(cleanup_ids, user_id, balance="20000.00", empire_id=empire["id"], account_type="PROP_FIRM")
    standalone_id = _create_account(cleanup_ids, user_id, balance="5000.00")

    allocation_id = _create_allocation(cleanup_ids, standalone_id)
    client.post(f"/allocations/{allocation_id}/confirm")

    note = client.post(
        "/journal-notes", json={"related_allocation_id": allocation_id, "text": "Dashboard-Test"}
    ).json()
    cleanup_ids["note"].append(note["id"])

    response = client.get(f"/users/{user_id}/portfolio")
    assert response.status_code == 200
    body = response.json()

    assert body["total_balance"] == "25000.00"
    assert len(body["empires"]) == 1
    assert body["empires"][0]["total_balance"] == "20000.00"
    assert len(body["standalone_accounts"]) == 1
    assert body["standalone_accounts"][0]["account_id"] == standalone_id
    assert body["active_trade_count"] == 1
    assert any(entry["allocation_id"] == allocation_id for entry in body["recent_journal_entries"])


def test_portfolio_for_user_without_data_returns_zeroed_dashboard(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)

    response = client.get(f"/users/{user_id}/portfolio")
    assert response.status_code == 200
    body = response.json()
    assert body["total_balance"] == "0"
    assert body["empires"] == []
    assert body["standalone_accounts"] == []
    assert body["active_trade_count"] == 0
    assert body["recent_journal_entries"] == []


def test_active_trades_filter_combines_confirmed_and_open_across_accounts(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    account_a = _create_account(cleanup_ids, user_id)
    account_b = _create_account(cleanup_ids, user_id)

    created_alloc = _create_allocation(cleanup_ids, account_a, pair="EURUSD")
    confirmed_alloc = _create_allocation(cleanup_ids, account_b, pair="GBPUSD")
    client.post(f"/allocations/{confirmed_alloc}/confirm")
    opened_alloc = _create_allocation(cleanup_ids, account_a, pair="AUDUSD")
    client.post(f"/allocations/{opened_alloc}/confirm")
    client.post(f"/allocations/{opened_alloc}/open", json={})

    response = client.get("/allocations", params={"user_id": user_id, "status": "ACTIVE"})
    assert response.status_code == 200
    ids = {row["allocation_id"] for row in response.json()}
    assert ids == {confirmed_alloc, opened_alloc}
    assert created_alloc not in ids


def test_journal_user_id_filter_covers_all_accounts(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    account_a = _create_account(cleanup_ids, user_id)
    account_b = _create_account(cleanup_ids, user_id)

    alloc_a = _create_allocation(cleanup_ids, account_a, pair="EURUSD")
    alloc_b = _create_allocation(cleanup_ids, account_b, pair="USDJPY")

    response = client.get("/journal", params={"user_id": user_id})
    assert response.status_code == 200
    allocation_ids = {row["allocation_id"] for row in response.json()}
    assert {alloc_a, alloc_b}.issubset(allocation_ids)


def test_empire_accounts_lists_only_members(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    empire = client.post("/empires", json={"user_id": user_id, "name": "Member-Test"}).json()
    cleanup_ids["empire"].append(empire["id"])

    member_id = _create_account(cleanup_ids, user_id, balance="15000.00", empire_id=empire["id"])
    _create_account(cleanup_ids, user_id, balance="9000.00")  # standalone, darf nicht auftauchen

    response = client.get(f"/empires/{empire['id']}/accounts")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["account_id"] == member_id
    assert body[0]["balance"] == "15000.00"
