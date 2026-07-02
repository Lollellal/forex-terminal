"""core.trade_allocations (minimal, Implementierungsschritt 3)

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-02

Baut das TradeAllocation-Fundament aus Implementierungsschritt 3
(BACKEND_ARCHITECTURE.md §2.1/§2.2). Der Status-Lifecycle ist bewusst auf
CREATED -> CONFIRMED -> OPEN -> CLOSED verkürzt gegenüber dem vollen
Zielschema (PROPOSED/PARTIAL_HIT/BREAK_EVEN/CANCELLED) — diese Zustände
gehören fachlich zum Risk Gate bzw. zur Async-Ausführungslogik, die hier
laut Auftrag explizit noch nicht gebaut wird. Additive Migration erweitert
den CHECK-Constraint, sobald diese Schritte kommen.

signal_id ist bewusst ohne Foreign Key (core.signals existiert noch nicht,
Signal-Aggregate ist nicht Teil dieses Schritts) — nur ein Platzhalter für
die spätere Signal-Verknüpfung.
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE core.trade_allocations (
            id                   UUID PRIMARY KEY,
            account_id           UUID NOT NULL REFERENCES core.accounts(id),
            signal_id            UUID NULL,
            pair                 TEXT NOT NULL,
            direction            TEXT NOT NULL CHECK (direction IN ('LONG','SHORT')),
            status               TEXT NOT NULL CHECK (status IN
                                 ('CREATED','CONFIRMED','OPEN','CLOSED')),
            planned_risk_pct     NUMERIC(5,2) NOT NULL,
            entry_price_planned  NUMERIC(12,5) NULL,
            sl_price             NUMERIC(12,5) NULL,
            tp_price             NUMERIC(12,5) NULL,
            fill_price           NUMERIC(12,5) NULL,
            fill_time            TIMESTAMPTZ NULL,
            closed_at            TIMESTAMPTZ NULL,
            close_reason         TEXT NULL CHECK (close_reason IN
                                 ('SL','TP','TIME_EXIT','FORCE_CLOSE','MANUAL') OR close_reason IS NULL),
            realized_r           NUMERIC(6,3) NULL,
            version              INTEGER NOT NULL DEFAULT 0,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_allocations_account_status "
        "ON core.trade_allocations (account_id, status)"
    )
    op.execute(
        "CREATE INDEX idx_allocations_pair_status ON core.trade_allocations (pair, status)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS core.trade_allocations")
