"""core.weekly_reports (Implementierungsschritt 8)

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-02

Registriert bereits vom Desktop generierte Weekly Reports in der zentralen
DB (BACKEND_ARCHITECTURE.md §2.1) — keine Report-Generierung hier, nur
Persistenz von Metadaten + content_ref (Supabase-Storage-Pfad). Status-Enum
übernimmt bewusst das volle Zielschema (DRAFT/GENERATED/PUBLISHED/ARCHIVED),
auch wenn dieser Schritt nur GENERATED->PUBLISHED erzeugt — DRAFT/ARCHIVED
sind reine Enum-Werte ohne Zusatzaufwand, spart eine spätere Migration.
"""

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE core.weekly_reports (
            id             UUID PRIMARY KEY,
            user_id        UUID NOT NULL REFERENCES core.users(id),
            period_start   DATE NOT NULL,
            period_end     DATE NOT NULL,
            status         TEXT NOT NULL CHECK (status IN ('DRAFT','GENERATED','PUBLISHED','ARCHIVED')),
            content_ref    TEXT NULL,
            published_at   TIMESTAMPTZ NULL,
            version        INTEGER NOT NULL DEFAULT 0,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_weekly_reports_user ON core.weekly_reports (user_id, period_start)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS core.weekly_reports")
