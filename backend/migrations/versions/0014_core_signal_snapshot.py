"""core: signal_snapshot auf trade_allocations, summary auf weekly_reports (v1.1 Mobile Trade Intelligence)

Revision ID: 0014
Revises: 0013
Create Date: 2026-07-02

Beide Spalten sind additiv und werden ausschließlich beim jeweiligen
CREATE-Event gesetzt (ALLOCATION_CREATED / WEEKLY_REPORT_GENERATED) — danach
nie mehr verändert, echte Unveränderlichkeit statt nur Konvention.
signal_snapshot ist bewusst JSONB statt Einzelspalten: der Desktop-Publish
soll künftig weitere strukturierte Signal-Felder ergänzen können, ohne
weitere Migrationen (siehe src/trade_snapshot.py).
"""

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE core.trade_allocations ADD COLUMN signal_snapshot JSONB NULL")
    op.execute("ALTER TABLE core.weekly_reports ADD COLUMN summary TEXT NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE core.weekly_reports DROP COLUMN IF EXISTS summary")
    op.execute("ALTER TABLE core.trade_allocations DROP COLUMN IF EXISTS signal_snapshot")
