"""Integrationstests gegen echtes PostgreSQL (Supabase) über den FastAPI
TestClient — Implementierungsschritt 6: API Layer.

Kein Mock der Domain-Schicht: jeder Request läuft durch dieselbe
Transaktions-/Projection-Pipeline wie die Domain-Tests aus den Schritten
2-5, nur über HTTP statt direktem Python-Aufruf.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.api.app import app
from backend.api.auth import get_api_auth_token

client = TestClient(app)
try:
    client.headers["Authorization"] = f"Bearer {get_api_auth_token()}"
except RuntimeError as exc:
    pytest.skip(str(exc), allow_module_level=True)


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
        conn.execute(text("INSERT INTO core.users (id, label) VALUES (:id, 'pytest-api')"), {"id": user_id})
    cleanup_ids["user"].append(user_id)
    return user_id


def test_create_account_and_read_balance_projection(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)

    response = client.post(
        "/accounts",
        json={
            "user_id": user_id,
            "account_type": "LIVE",
            "initial_balance": "10000.00",
            "initial_equity": "10000.00",
        },
    )
    assert response.status_code == 201
    body = response.json()
    cleanup_ids["account"].append(body["id"])
    assert body["status"] == "ACTIVE"
    assert body["balance"] == "10000.00"

    balance_response = client.get(f"/accounts/{body['id']}/balance")
    assert balance_response.status_code == 200
    assert balance_response.json()["balance"] == "10000.00"


def test_update_balance_via_api(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    account_id = client.post(
        "/accounts",
        json={
            "user_id": user_id,
            "account_type": "LIVE",
            "initial_balance": "5000.00",
            "initial_equity": "5000.00",
        },
    ).json()["id"]
    cleanup_ids["account"].append(account_id)

    response = client.post(
        f"/accounts/{account_id}/balance", json={"balance": "5250.00", "equity": "5180.00"}
    )
    assert response.status_code == 200
    assert response.json()["balance"] == "5250.00"
    assert response.json()["version"] == 2


def test_get_account_balance_404_when_unknown(db_engine, cleanup_ids):
    response = client.get(f"/accounts/{uuid.uuid4()}/balance")
    assert response.status_code == 404


def test_empire_overview_sums_accounts_via_api(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)

    empire = client.post("/empires", json={"user_id": user_id, "name": "API-Empire"}).json()
    cleanup_ids["empire"].append(empire["id"])

    for balance in ("10000.00", "20000.00"):
        account = client.post(
            "/accounts",
            json={
                "user_id": user_id,
                "account_type": "PROP_FIRM",
                "initial_balance": balance,
                "initial_equity": balance,
                "empire_id": empire["id"],
            },
        ).json()
        cleanup_ids["account"].append(account["id"])

    overview = client.get(f"/empires/{empire['id']}/overview")
    assert overview.status_code == 200
    body = overview.json()
    assert body["account_count"] == 2
    assert body["total_balance"] == "30000.00"


def test_allocation_full_lifecycle_via_api(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    account_id = client.post(
        "/accounts",
        json={
            "user_id": user_id,
            "account_type": "LIVE",
            "initial_balance": "10000.00",
            "initial_equity": "10000.00",
        },
    ).json()["id"]
    cleanup_ids["account"].append(account_id)

    created = client.post(
        "/allocations",
        json={
            "account_id": account_id,
            "pair": "EURUSD",
            "direction": "LONG",
            "planned_risk_pct": "1.0",
            "entry_price_planned": "1.08500",
        },
    )
    assert created.status_code == 201
    allocation_id = created.json()["id"]
    cleanup_ids["allocation"].append(allocation_id)
    assert created.json()["status"] == "CREATED"

    confirmed = client.post(f"/allocations/{allocation_id}/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "CONFIRMED"
    assert confirmed.json()["applied_risk_pct"] == "1.0"

    opened = client.post(f"/allocations/{allocation_id}/open", json={})
    assert opened.status_code == 200
    assert opened.json()["status"] == "OPEN"

    closed = client.post(
        f"/allocations/{allocation_id}/close", json={"close_reason": "TP", "realized_r": "1.8"}
    )
    assert closed.status_code == 200
    assert closed.json()["status"] == "CLOSED"
    assert closed.json()["realized_r"] == "1.8"

    listing = client.get("/allocations", params={"account_id": account_id, "status": "CLOSED"})
    assert listing.status_code == 200
    assert len(listing.json()) == 1
    assert listing.json()[0]["allocation_id"] == allocation_id

    single = client.get(f"/allocations/{allocation_id}")
    assert single.status_code == 200
    assert single.json()["realized_r"] == "1.800"


def test_closing_allocation_recomputes_account_balance(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    account_id = client.post(
        "/accounts",
        json={
            "user_id": user_id,
            "account_type": "PROP_FIRM",
            "initial_balance": "10000.00",
            "initial_equity": "10000.00",
        },
    ).json()["id"]
    cleanup_ids["account"].append(account_id)

    def _open_and_close(pair: str, realized_r: str) -> None:
        allocation_id = client.post(
            "/allocations",
            json={"account_id": account_id, "pair": pair, "direction": "LONG", "planned_risk_pct": "1.0"},
        ).json()["id"]
        cleanup_ids["allocation"].append(allocation_id)
        client.post(f"/allocations/{allocation_id}/confirm")
        client.post(f"/allocations/{allocation_id}/open", json={})
        client.post(f"/allocations/{allocation_id}/close", json={"close_reason": "TP", "realized_r": realized_r})

    # WIN: +1.7R auf 1% Risiko -> +170; LOSS: -1R auf 1% Risiko -> -100. Netto +70.
    _open_and_close("EURUSD", "1.7")
    _open_and_close("GBPUSD", "-1.0")

    balance = client.get(f"/accounts/{account_id}/balance")
    assert balance.status_code == 200
    assert balance.json()["balance"] == "10070.00"
    assert balance.json()["equity"] == "10070.00"


def test_create_allocation_invalid_direction_returns_400(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    account_id = client.post(
        "/accounts",
        json={
            "user_id": user_id,
            "account_type": "LIVE",
            "initial_balance": "10000.00",
            "initial_equity": "10000.00",
        },
    ).json()["id"]
    cleanup_ids["account"].append(account_id)

    response = client.post(
        "/allocations",
        json={"account_id": account_id, "pair": "EURUSD", "direction": "SIDEWAYS", "planned_risk_pct": "1.0"},
    )
    assert response.status_code == 400


def test_confirm_unknown_allocation_returns_404(db_engine, cleanup_ids):
    response = client.post(f"/allocations/{uuid.uuid4()}/confirm")
    assert response.status_code == 404


def test_confirm_allocation_twice_returns_400(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    account_id = client.post(
        "/accounts",
        json={
            "user_id": user_id,
            "account_type": "LIVE",
            "initial_balance": "10000.00",
            "initial_equity": "10000.00",
        },
    ).json()["id"]
    cleanup_ids["account"].append(account_id)
    allocation_id = client.post(
        "/allocations",
        json={"account_id": account_id, "pair": "EURUSD", "direction": "LONG", "planned_risk_pct": "1.0"},
    ).json()["id"]
    cleanup_ids["allocation"].append(allocation_id)

    first = client.post(f"/allocations/{allocation_id}/confirm")
    assert first.status_code == 200
    second = client.post(f"/allocations/{allocation_id}/confirm")
    assert second.status_code == 400


def test_journal_note_via_api_does_not_change_trade_facts(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    account_id = client.post(
        "/accounts",
        json={
            "user_id": user_id,
            "account_type": "LIVE",
            "initial_balance": "10000.00",
            "initial_equity": "10000.00",
        },
    ).json()["id"]
    cleanup_ids["account"].append(account_id)
    allocation_id = client.post(
        "/allocations",
        json={"account_id": account_id, "pair": "EURUSD", "direction": "LONG", "planned_risk_pct": "1.0"},
    ).json()["id"]
    cleanup_ids["allocation"].append(allocation_id)

    before = client.get(f"/journal/{allocation_id}").json()

    note_response = client.post(
        "/journal-notes",
        json={"related_allocation_id": allocation_id, "text": "API-Test-Notiz", "attachments": []},
    )
    assert note_response.status_code == 201
    cleanup_ids["note"].append(note_response.json()["id"])

    after = client.get(f"/journal/{allocation_id}").json()
    assert after["status"] == before["status"]
    assert after["planned_risk_pct"] == before["planned_risk_pct"]
    assert after["realized_r"] == before["realized_r"]
    assert len(after["notes"]) == 1
    assert after["notes"][0]["text"] == "API-Test-Notiz"
