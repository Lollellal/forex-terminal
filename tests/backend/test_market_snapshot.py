"""Integrationstest fuer den Market-Snapshot-Endpunkt (v1.1 Mobile Trade
Intelligence). Kein Aggregate/Event-Sourcing hier — projections.market_snapshot
ist ein reiner, direkt geschriebener Cache-Wert (siehe Migration 0015)."""

from __future__ import annotations

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
def clear_market_snapshot(db_engine):
    """Sichert eine evtl. bereits vorhandene 'current'-Zeile und stellt sie
    nach dem Test wieder her — projections.market_snapshot ist ein Singleton
    ohne eigene Test-Isolation (keine zufällige UUID wie bei anderen
    Fixtures), ein simples DELETE würde sonst echte, vom Desktop-Publish
    geschriebene Produktionsdaten zerstören."""
    with db_engine.begin() as conn:
        original = conn.execute(
            text("SELECT regime, vix, yield_curve, updated_at FROM projections.market_snapshot WHERE id = 'current'")
        ).mappings().first()
        conn.execute(text("DELETE FROM projections.market_snapshot WHERE id = 'current'"))
    yield
    with db_engine.begin() as conn:
        conn.execute(text("DELETE FROM projections.market_snapshot WHERE id = 'current'"))
        if original is not None:
            conn.execute(
                text(
                    """
                    INSERT INTO projections.market_snapshot (id, regime, vix, yield_curve, updated_at)
                    VALUES ('current', :regime, :vix, :yield_curve, :updated_at)
                    """
                ),
                dict(original),
            )


def test_returns_404_when_no_snapshot_published(db_engine, clear_market_snapshot):
    response = client.get("/market-snapshot")
    assert response.status_code == 404


def test_returns_current_snapshot_when_published(db_engine, clear_market_snapshot):
    with db_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO projections.market_snapshot (id, regime, vix, yield_curve, updated_at)
                VALUES ('current', 'TREND', 18.4, 0.31, now())
                """
            )
        )

    response = client.get("/market-snapshot")
    assert response.status_code == 200
    body = response.json()
    assert body["regime"] == "TREND"
    assert body["vix"] == "18.4" or float(body["vix"]) == 18.4
