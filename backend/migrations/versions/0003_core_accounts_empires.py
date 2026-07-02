"""core.empires + core.accounts (minimal, Implementierungsschritt 2)

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-02

Baut ausschließlich das Account/Empire-Fundament aus Implementierungsschritt 2
(BACKEND_ARCHITECTURE.md §2.1/§2.2). Bewusst ohne prop_firm_template_id, stage
und risk_*-Spalten aus dem vollen Zielschema — die gehören fachlich zum
Risk-Gate-/Prop-Firm-Schritt, der hier explizit nicht im Scope ist. Additive
Migration ergänzt diese Spalten, sobald der Risk Gate gebaut wird.
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE core.empires (
            id            UUID PRIMARY KEY,
            user_id       UUID NOT NULL REFERENCES core.users(id),
            name          TEXT NOT NULL,
            version       INTEGER NOT NULL DEFAULT 0,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE core.accounts (
            id            UUID PRIMARY KEY,
            user_id       UUID NOT NULL REFERENCES core.users(id),
            empire_id     UUID NULL REFERENCES core.empires(id),
            account_type  TEXT NOT NULL CHECK (account_type IN ('LIVE','PROP_FIRM')),
            status        TEXT NOT NULL CHECK (status IN ('ACTIVE','CLOSED')),
            balance       NUMERIC(14,2) NOT NULL,
            equity        NUMERIC(14,2) NOT NULL,
            version       INTEGER NOT NULL DEFAULT 0,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_accounts_empire ON core.accounts (empire_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS core.accounts")
    op.execute("DROP TABLE IF EXISTS core.empires")
