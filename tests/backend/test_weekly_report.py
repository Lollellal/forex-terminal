"""Integrationstests gegen echtes PostgreSQL (Supabase) + echtes Supabase
Storage — Implementierungsschritt 8: WeeklyReport Persistence.

Definition of Done: ein bereits generiertes PDF lässt sich registrieren
(Upload nach Storage + Metadaten in core/projections), veröffentlichen, über
GET /users/{id}/weekly-reports auflisten und über eine Signed URL laden.
Keine Report-Generierung, keine KI-Logik hier — nur Persistenz.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.api.app import app
from backend.infrastructure.storage import WeeklyReportStorage

client = TestClient(app)


@pytest.fixture
def cleanup_ids(db_engine):
    ids = {"user": [], "report": []}
    yield ids
    storage = WeeklyReportStorage()
    with db_engine.begin() as conn:
        for report_id in ids["report"]:
            row = conn.execute(
                text("SELECT content_ref FROM core.weekly_reports WHERE id = :id"), {"id": report_id}
            ).first()
            if row and row[0]:
                try:
                    storage.delete(row[0])
                except Exception:
                    pass
            conn.execute(text("DELETE FROM projections.weekly_reports WHERE id = :id"), {"id": report_id})
            conn.execute(text("DELETE FROM core.weekly_reports WHERE id = :id"), {"id": report_id})
            conn.execute(
                text(
                    "DELETE FROM core.event_store WHERE aggregate_type = 'WeeklyReport' AND aggregate_id = :id"
                ),
                {"id": report_id},
            )
        for uid in ids["user"]:
            conn.execute(text("DELETE FROM core.users WHERE id = :id"), {"id": uid})


def _make_user(db_engine, cleanup_ids) -> str:
    user_id = str(uuid.uuid4())
    with db_engine.begin() as conn:
        conn.execute(
            text("INSERT INTO core.users (id, label) VALUES (:id, 'pytest-weekly-report')"),
            {"id": user_id},
        )
    cleanup_ids["user"].append(user_id)
    return user_id


def _register(
    cleanup_ids,
    user_id,
    period_start="2026-06-22",
    period_end="2026-06-28",
    content=b"%PDF-fake-content",
    summary=None,
):
    data = {"user_id": user_id, "period_start": period_start, "period_end": period_end}
    if summary is not None:
        data["summary"] = summary
    response = client.post(
        "/weekly-reports",
        data=data,
        files={"file": ("report.pdf", content, "application/pdf")},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    cleanup_ids["report"].append(body["id"])
    return body


def test_register_uploads_to_storage_and_persists_metadata(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)

    body = _register(cleanup_ids, user_id)

    assert body["status"] == "GENERATED"
    assert body["user_id"] == user_id
    assert body["period_start"] == "2026-06-22"
    assert body["content_ref"] == f"{user_id}/2026-06-22_2026-06-28.pdf"
    assert body["published_at"] is None

    with db_engine.connect() as conn:
        state_row = conn.execute(
            text("SELECT status, content_ref, version FROM core.weekly_reports WHERE id = :id"),
            {"id": body["id"]},
        ).one()
        projection_row = conn.execute(
            text("SELECT status, content_ref FROM projections.weekly_reports WHERE id = :id"),
            {"id": body["id"]},
        ).one()

    assert state_row.status == "GENERATED"
    assert state_row.version == 1
    assert projection_row.status == "GENERATED"
    assert projection_row.content_ref == body["content_ref"]


def test_uploaded_pdf_is_actually_retrievable_from_storage(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    original_content = b"%PDF-1.4 real bytes for roundtrip check"
    body = _register(cleanup_ids, user_id, content=original_content)

    storage = WeeklyReportStorage()
    downloaded = storage._client.storage.from_("weekly-reports").download(body["content_ref"])
    assert downloaded == original_content


def test_publish_transitions_status_and_sets_published_at(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    body = _register(cleanup_ids, user_id)

    response = client.post(f"/weekly-reports/{body['id']}/publish")
    assert response.status_code == 200
    published = response.json()
    assert published["status"] == "PUBLISHED"
    assert published["published_at"] is not None

    with db_engine.connect() as conn:
        projection_row = conn.execute(
            text("SELECT status, published_at FROM projections.weekly_reports WHERE id = :id"),
            {"id": body["id"]},
        ).one()
    assert projection_row.status == "PUBLISHED"
    assert projection_row.published_at is not None


def test_publish_twice_returns_400(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    body = _register(cleanup_ids, user_id)
    client.post(f"/weekly-reports/{body['id']}/publish")

    second = client.post(f"/weekly-reports/{body['id']}/publish")
    assert second.status_code == 400


def test_download_url_returns_working_signed_url(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    body = _register(cleanup_ids, user_id)

    response = client.get(f"/weekly-reports/{body['id']}/download-url")
    assert response.status_code == 200
    payload = response.json()
    assert payload["url"].startswith("https://")
    assert body["content_ref"] in payload["url"]


def test_register_persists_summary_and_defaults_to_none(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)

    with_summary = _register(cleanup_ids, user_id, summary="Der Schweizer Franken ist die staerkste Waehrung.")
    assert with_summary["summary"] == "Der Schweizer Franken ist die staerkste Waehrung."

    without_summary = _register(
        cleanup_ids, user_id, period_start="2026-06-15", period_end="2026-06-21"
    )
    assert without_summary["summary"] is None

    with db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT summary FROM projections.weekly_reports WHERE id = :id"),
            {"id": with_summary["id"]},
        ).one()
    assert row.summary == "Der Schweizer Franken ist die staerkste Waehrung."


def test_list_weekly_reports_for_user(db_engine, cleanup_ids):
    user_id = _make_user(db_engine, cleanup_ids)
    first = _register(cleanup_ids, user_id, period_start="2026-06-15", period_end="2026-06-21")
    second = _register(cleanup_ids, user_id, period_start="2026-06-22", period_end="2026-06-28")

    response = client.get(f"/users/{user_id}/weekly-reports")
    assert response.status_code == 200
    ids = [row["id"] for row in response.json()]
    assert first["id"] in ids
    assert second["id"] in ids
    # neuestes zuerst
    assert ids.index(second["id"]) < ids.index(first["id"])
