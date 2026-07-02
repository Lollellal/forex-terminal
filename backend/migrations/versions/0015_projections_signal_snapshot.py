"""projections: signal_snapshot/summary spiegeln, neue market_snapshot-Tabelle (v1.1 Mobile Trade Intelligence)

Revision ID: 0015
Revises: 0014
Create Date: 2026-07-02

market_snapshot ist bewusst KEIN Event-Sourced-Aggregate (kein core-Pendant,
keine Business-Regeln) — nur ein einzeiliger, vom Desktop-Publish direkt
upgeserteter Cache-Wert des zuletzt bekannten Marktregimes, analog zum
account_snapshot-Pattern in projections.journal_view.
"""

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE projections.allocation_overview ADD COLUMN signal_snapshot JSONB NULL")
    op.execute("ALTER TABLE projections.allocation_overview ADD COLUMN opened_at TIMESTAMPTZ NULL")
    op.execute("ALTER TABLE projections.weekly_reports ADD COLUMN summary TEXT NULL")
    op.execute(
        """
        CREATE TABLE projections.market_snapshot (
            id           TEXT PRIMARY KEY DEFAULT 'current',
            regime       TEXT NOT NULL,
            vix          NUMERIC(6,2) NULL,
            yield_curve  NUMERIC(6,3) NULL,
            updated_at   TIMESTAMPTZ NOT NULL
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS projections.market_snapshot")
    op.execute("ALTER TABLE projections.weekly_reports DROP COLUMN IF EXISTS summary")
    op.execute("ALTER TABLE projections.allocation_overview DROP COLUMN IF EXISTS opened_at")
    op.execute("ALTER TABLE projections.allocation_overview DROP COLUMN IF EXISTS signal_snapshot")
