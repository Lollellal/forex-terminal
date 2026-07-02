"""projections.weekly_reports (Implementierungsschritt 8)

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-02

Mobile liest wie alle anderen Clients aus projections, nie direkt aus core
(BACKEND_ARCHITECTURE.md §2.5/§2.6) — auch wenn die Zeilenform hier fast
1:1 dem State entspricht, bleibt die Trennung konsistent (core bleibt
service-intern, projections ist der Client-Lesepfad, jederzeit per Replay
neu aufbaubar).
"""

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE projections.weekly_reports (
            id             UUID PRIMARY KEY,
            user_id        UUID NOT NULL,
            period_start   DATE NOT NULL,
            period_end     DATE NOT NULL,
            status         TEXT NOT NULL,
            content_ref    TEXT NULL,
            published_at   TIMESTAMPTZ NULL,
            updated_at     TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX idx_proj_weekly_reports_user ON projections.weekly_reports (user_id, period_start)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS projections.weekly_reports")
