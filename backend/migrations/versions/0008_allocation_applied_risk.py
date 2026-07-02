"""applied_risk_pct auf trade_allocations + Projection (Risk-Gate-Ergebnis)

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-02

Additive Ergänzung, angekündigt im Docstring von 0005_core_trade_allocations:
Der Risk Gate kann das geplante Risiko bei Confirm/Open anpassen —
applied_risk_pct hält das tatsächlich freigegebene Risiko fest, getrennt von
planned_risk_pct (dem ursprünglichen Wunsch beim Erstellen der Allocation).
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE core.trade_allocations ADD COLUMN applied_risk_pct NUMERIC(5,2) NULL")
    op.execute(
        "ALTER TABLE projections.allocation_overview ADD COLUMN applied_risk_pct NUMERIC(5,2) NULL"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE projections.allocation_overview DROP COLUMN IF EXISTS applied_risk_pct")
    op.execute("ALTER TABLE core.trade_allocations DROP COLUMN IF EXISTS applied_risk_pct")
