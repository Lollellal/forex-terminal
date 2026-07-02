"""Fill-Preis aus dem Domain-Modell entfernen (fachliche Korrektur)

Revision ID: 0013
Revises: 0012
Create Date: 2026-07-02

Dieses System ist kein Order-Management-System, sondern ein Decision-/
Research-/Performance-System — die fachliche Wahrheit ist "ich habe
beschlossen, dieses Signal auf diesem Account zu handeln", nicht "der
Broker hat zu Preis X gefillt". Der exakte Fill-Preis hat für Performance,
Risk Management, Journal, Empire, Reports und Mobile keinen Mehrwert
(Performance basiert ausschließlich auf Pair/Richtung/Account/Risiko%/
Status/realized_r/close_reason/Zeitpunkten). fill_time bleibt als Zeitpunkt
erhalten, wird aber zu opened_at umbenannt — er markiert weiterhin "wann
wurde diese Allocation aktiv", nur ohne die Broker-Fill-Konnotation.

Keine Datenmigration nötig: core.event_store enthält zum Zeitpunkt dieser
Migration keine einzige AllocationOpened-Zeile (geprüft vor dem Schreiben
dieser Migration) — reine Schema-Änderung.
"""

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE core.trade_allocations DROP COLUMN fill_price")
    op.execute("ALTER TABLE core.trade_allocations RENAME COLUMN fill_time TO opened_at")
    op.execute("ALTER TABLE projections.allocation_overview DROP COLUMN fill_price")
    op.execute("ALTER TABLE projections.journal_view DROP COLUMN fill_price")


def downgrade() -> None:
    op.execute("ALTER TABLE projections.journal_view ADD COLUMN fill_price NUMERIC(12,5) NULL")
    op.execute("ALTER TABLE projections.allocation_overview ADD COLUMN fill_price NUMERIC(12,5) NULL")
    op.execute("ALTER TABLE core.trade_allocations RENAME COLUMN opened_at TO fill_time")
    op.execute("ALTER TABLE core.trade_allocations ADD COLUMN fill_price NUMERIC(12,5) NULL")
