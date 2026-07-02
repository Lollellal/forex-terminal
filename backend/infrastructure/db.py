"""DB-Engine-Erzeugung aus DATABASE_URL (Supabase-Connection-URI in .env).

Siehe BACKEND_ARCHITECTURE.md §2.1. Treiber ist psycopg (v3) — die
Normalisierung erlaubt sowohl die von Supabase kopierte
``postgresql://...``-URI als auch ``postgres://...`` unverändert in .env.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL ist nicht gesetzt. Trage die Supabase-Connection-URI "
            "(Settings -> Database -> Connection string) als DATABASE_URL in .env ein "
            "(siehe BACKEND_ARCHITECTURE.md §2.1)."
        )
    return normalize_database_url(url)


def get_engine() -> Engine:
    return create_engine(get_database_url(), pool_pre_ping=True)
