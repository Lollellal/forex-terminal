"""minimal core.users stub

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-02

Nur ein FK-Ziel für accounts.user_id / empires.user_id (BACKEND_ARCHITECTURE.md
§2.1, Entscheidung 3). Kein Auth, kein Login, keine Sessions/Devices — das
bleibt bewusst einem späteren Auth-Schritt vorbehalten. Single-User-System,
daher genügt hier eine schlanke Tabelle statt des vollen users/devices/sessions-
Blocks aus der Implementierungsreihenfolge.
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE core.users (
            id            UUID PRIMARY KEY,
            label         TEXT NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS core.users")
