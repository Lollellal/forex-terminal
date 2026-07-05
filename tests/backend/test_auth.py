"""Tests für die globale Bearer-Token-Auth (backend/api/auth.py, Phase 1
"PC muss nicht mehr an sein"). Da die Dependency app-weit registriert ist
(siehe app.py), reicht es, den Mechanismus stellvertretend an zwei
Endpunkten unterschiedlicher Router zu prüfen — jede Route hängt an
derselben globalen Dependency, keine Route-für-Route-Duplizierung nötig.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.auth import get_api_auth_token

try:
    _TOKEN = get_api_auth_token()
except RuntimeError as exc:
    pytest.skip(str(exc), allow_module_level=True)

client = TestClient(app)

_PROTECTED_PATHS = ["/market-snapshot", "/allocations"]


@pytest.mark.parametrize("path", _PROTECTED_PATHS)
def test_request_without_token_is_rejected(path):
    response = client.get(path)
    assert response.status_code == 401


@pytest.mark.parametrize("path", _PROTECTED_PATHS)
def test_request_with_wrong_token_is_rejected(path):
    response = client.get(path, headers={"Authorization": "Bearer not-the-real-token"})
    assert response.status_code == 401


@pytest.mark.parametrize("path", _PROTECTED_PATHS)
def test_request_with_correct_token_passes_auth(path):
    response = client.get(path, headers={"Authorization": f"Bearer {_TOKEN}"})
    # Auth ist bestanden — ein evtl. 404 (kein Market-Snapshot veröffentlicht)
    # ist ein fachliches Ergebnis, kein Auth-Fehler. Nur 401 wäre falsch.
    assert response.status_code != 401


def test_request_with_malformed_header_is_rejected():
    response = client.get("/market-snapshot", headers={"Authorization": _TOKEN})
    assert response.status_code == 401
