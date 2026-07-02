"""projections schema: account_balances, empire_overview, checkpoints

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-02

Nur die zwei Projektionen aus Implementierungsschritt 2 (BACKEND_ARCHITECTURE.md
§2.1 "Projections"). Keine Foreign Keys zu core — Projektionen sind jederzeit
per TRUNCATE + Replay aus core.event_store neu aufbaubar.
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS projections")

    op.execute(
        """
        CREATE TABLE projections.checkpoints (
            projection_name     TEXT PRIMARY KEY,
            last_processed_seq  BIGINT NOT NULL DEFAULT 0
        )
        """
    )

    op.execute(
        """
        CREATE TABLE projections.account_balances (
            account_id    UUID PRIMARY KEY,
            empire_id     UUID NULL,
            account_type  TEXT NOT NULL,
            status        TEXT NOT NULL,
            balance       NUMERIC(14,2) NOT NULL,
            equity        NUMERIC(14,2) NOT NULL,
            updated_at    TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_projections_account_balances_empire "
        "ON projections.account_balances (empire_id)"
    )

    op.execute(
        """
        CREATE TABLE projections.empire_overview (
            empire_id       UUID PRIMARY KEY,
            name            TEXT NOT NULL,
            account_count   INTEGER NOT NULL DEFAULT 0,
            total_balance   NUMERIC(14,2) NOT NULL DEFAULT 0,
            total_equity    NUMERIC(14,2) NOT NULL DEFAULT 0,
            updated_at      TIMESTAMPTZ NOT NULL
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS projections.empire_overview")
    op.execute("DROP TABLE IF EXISTS projections.account_balances")
    op.execute("DROP TABLE IF EXISTS projections.checkpoints")
    op.execute("DROP SCHEMA IF EXISTS projections CASCADE")
