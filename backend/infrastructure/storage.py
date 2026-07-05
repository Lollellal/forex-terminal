"""Supabase-Storage-Client für Weekly-Report-PDFs (Implementierungsschritt 8).

Nur für Weekly Reports — kein allgemeiner Datei-Storage-Layer. Der Desktop
generiert das PDF (bestehende Pipeline, unverändert), dieser Client lädt es
in einen privaten Bucket hoch und liefert den Storage-Pfad, der als
content_ref in core.weekly_reports landet. Mobile bekommt später eine
zeitlich befristete Signed URL zum Lesen — der Bucket ist bewusst privat,
damit PDFs nicht öffentlich erraten werden können.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

_BUCKET = "weekly-reports"
_SIGNED_URL_TTL_SECONDS = 3600


class WeeklyReportStorageError(Exception):
    """SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY fehlen oder der Storage-Call
    ist fehlgeschlagen."""


def _get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise WeeklyReportStorageError(
            "SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY sind nicht gesetzt — siehe .env "
            "(Supabase Dashboard -> Settings -> API -> service_role secret)."
        )
    # .strip(): siehe backend/infrastructure/db.py get_database_url() — Hosting-
    # Dashboards übernehmen beim Einfügen gerne einen Zeilenumbruch mit.
    return create_client(url.strip(), key.strip())


class WeeklyReportStorage:
    def __init__(self, client: Client | None = None) -> None:
        self._client = client or _get_client()

    def upload(self, *, storage_path: str, content: bytes, content_type: str = "application/pdf") -> str:
        """Lädt das PDF hoch, überschreibt eine bestehende Datei am selben
        Pfad (upsert) — Registrierung ist idempotent bei Retry. Gibt den
        Storage-Pfad zurück, der als content_ref persistiert wird."""
        self._client.storage.from_(_BUCKET).upload(
            storage_path, content, file_options={"content-type": content_type, "upsert": "true"}
        )
        return storage_path

    def signed_url(self, storage_path: str, expires_in: int = _SIGNED_URL_TTL_SECONDS) -> str:
        result = self._client.storage.from_(_BUCKET).create_signed_url(storage_path, expires_in)
        return result["signedURL"]

    def delete(self, storage_path: str) -> None:
        self._client.storage.from_(_BUCKET).remove([storage_path])
