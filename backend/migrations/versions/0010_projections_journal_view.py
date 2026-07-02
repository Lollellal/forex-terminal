"""projections.journal_view (Implementierungsschritt 5)

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-02

Journal ist reine Projection, keine eigene Wahrheit (BACKEND_ARCHITECTURE.md
§2.1) — jederzeit per TRUNCATE + Replay aus core.event_store neu aufbaubar,
daher keine Foreign Keys. Eine Zeile pro Allocation (allocation_id als PK) —
der SIGNAL-Track aus dem vollen Zielschema (projections.journal_view mit
track-Spalte) ist bewusst zurückgestellt, da noch kein Signal-Aggregate
existiert. account_snapshot ist ein einmaliger Snapshot bei AllocationCreated
(account_type/balance/equity) — spätere AccountUpdated-Events verändern
bestehende Journal-Zeilen nicht mehr (Journal = Historie, nicht Live-Ansicht
des Accounts).
"""

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE projections.journal_view (
            allocation_id     UUID PRIMARY KEY,
            account_id        UUID NOT NULL,
            pair              TEXT NOT NULL,
            direction         TEXT NOT NULL,
            status            TEXT NOT NULL,
            planned_risk_pct  NUMERIC(5,2) NOT NULL,
            applied_risk_pct  NUMERIC(5,2) NULL,
            fill_price        NUMERIC(12,5) NULL,
            closed_at         TIMESTAMPTZ NULL,
            close_reason      TEXT NULL,
            realized_r        NUMERIC(6,3) NULL,
            account_snapshot  JSONB NOT NULL,
            notes             JSONB NOT NULL DEFAULT '[]',
            updated_at        TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX idx_journal_view_account ON projections.journal_view (account_id)")
    op.execute("CREATE INDEX idx_journal_view_status ON projections.journal_view (status)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS projections.journal_view")
