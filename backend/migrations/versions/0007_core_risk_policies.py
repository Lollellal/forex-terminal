"""core.risk_policies + Seed für die zwei Schritt-4-Policies

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-02

Persistiertes Policy-Fundament (BACKEND_ARCHITECTURE.md §2.1), bewusst ohne
action_type/trigger_events-Spalten aus dem vollen Zielschema — die
Entscheidungslogik (ALLOW/ADJUST/REJECT) lebt in den Python-Policy-Klassen
(domain/risk/policies/), core.risk_policies ist reine Registry/Konfiguration
(enabled, scope, priority, adjusted_risk_pct-Parameter). Das ist bewusst kein
generischer Rule-Engine-Ansatz — nur so viel Persistenz wie für
policy_key/priority/enabled/scope/evaluation_mode gebraucht wird.

Nur zwei Policies aktiv (same-pair-open, consecutive-losses), beide GLOBAL
skopiert. Keine Correlation-/News-/Daily-DD-/Prop-Firm-Policies.
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE core.risk_policies (
            id                TEXT PRIMARY KEY,
            policy_key        TEXT NOT NULL,
            name              TEXT NOT NULL,
            description       TEXT NOT NULL,
            evaluation_mode    TEXT NOT NULL CHECK (evaluation_mode IN ('SYNC_GATE','ASYNC_MONITOR')),
            scope_type        TEXT NOT NULL CHECK (scope_type IN
                               ('GLOBAL','USER','EMPIRE','PROP_FIRM','ACCOUNT','SIGNAL','ALLOCATION')),
            scope_id          TEXT NULL,
            priority          INTEGER NOT NULL,
            enabled           BOOLEAN NOT NULL DEFAULT TRUE,
            adjusted_risk_pct NUMERIC(5,2) NULL,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT chk_global_has_no_scope_id CHECK (
                (scope_type = 'GLOBAL' AND scope_id IS NULL) OR
                (scope_type <> 'GLOBAL' AND scope_id IS NOT NULL)
            )
        )
        """
    )
    op.execute("CREATE INDEX idx_risk_policies_scope ON core.risk_policies (scope_type, scope_id)")
    op.execute("CREATE INDEX idx_risk_policies_key ON core.risk_policies (policy_key)")

    op.execute(
        """
        INSERT INTO core.risk_policies
            (id, policy_key, name, description, evaluation_mode, scope_type, scope_id,
             priority, enabled, adjusted_risk_pct)
        VALUES
            ('same-pair-open-global', 'same-pair-open', 'Same Pair Already Open',
             'Reduziert das Risiko, wenn auf diesem Account bereits eine bestätigte/offene '
             'Allocation auf demselben Pair existiert.',
             'SYNC_GATE', 'GLOBAL', NULL, 10, TRUE, 0.5),
            ('consecutive-losses-global', 'consecutive-losses', 'Two Consecutive Losses',
             'Reduziert das Risiko, wenn die letzten zwei geschlossenen Allocations dieses '
             'Accounts beide Verlust waren.',
             'SYNC_GATE', 'GLOBAL', NULL, 20, TRUE, 0.5)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS core.risk_policies")
