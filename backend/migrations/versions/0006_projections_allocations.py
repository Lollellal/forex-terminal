"""projections.allocation_overview (Implementierungsschritt 3)

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-02

Eine Projektion für aktive und geschlossene Allocations gemeinsam — Filter
über die status-Spalte (status <> 'CLOSED' = aktiv). Keine Foreign Keys zu
core, wie alle Projections (BACKEND_ARCHITECTURE.md §2.1) — jederzeit per
TRUNCATE + Replay neu aufbaubar.
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE projections.allocation_overview (
            allocation_id  UUID PRIMARY KEY,
            account_id     UUID NOT NULL,
            pair           TEXT NOT NULL,
            direction      TEXT NOT NULL,
            status         TEXT NOT NULL,
            planned_risk_pct NUMERIC(5,2) NOT NULL,
            fill_price     NUMERIC(12,5) NULL,
            closed_at      TIMESTAMPTZ NULL,
            close_reason   TEXT NULL,
            realized_r     NUMERIC(6,3) NULL,
            updated_at     TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_allocation_overview_account_status "
        "ON projections.allocation_overview (account_id, status)"
    )
    op.execute(
        "CREATE INDEX idx_allocation_overview_status "
        "ON projections.allocation_overview (status)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS projections.allocation_overview")
