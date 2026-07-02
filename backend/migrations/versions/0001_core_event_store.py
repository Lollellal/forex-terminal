"""core event store + idempotency foundation

Revision ID: 0001
Revises:
Create Date: 2026-07-01

Erste Migration des Trading-OS-Backends. Baut ausschließlich das Fundament
aus Implementierungsschritt 1: core.event_store + core.processed_commands.
Kein Empire, kein Account, kein Risk Gate, kein Mobile — siehe
BACKEND_ARCHITECTURE.md "Implementierungsreihenfolge".
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS core")
    # gen_random_uuid() für event_id-Defaults — auf Supabase i.d.R. bereits verfügbar.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.execute(
        """
        CREATE TABLE core.event_store (
            global_seq       BIGSERIAL PRIMARY KEY,
            event_id         UUID NOT NULL DEFAULT gen_random_uuid(),
            aggregate_type   TEXT NOT NULL,
            aggregate_id     UUID NOT NULL,
            version          INTEGER NOT NULL,
            event_type       TEXT NOT NULL,
            schema_version   INTEGER NOT NULL DEFAULT 1,
            payload          JSONB NOT NULL,
            source           TEXT NOT NULL
                             CHECK (source IN ('desktop','mobile','system','scheduled-job')),
            device_id        UUID NULL,
            correlation_id   UUID NOT NULL,
            causation_id     UUID NULL,
            occurred_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            recorded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_event_store_aggregate_version
                UNIQUE (aggregate_type, aggregate_id, version)
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_event_store_aggregate "
        "ON core.event_store (aggregate_type, aggregate_id, version)"
    )
    op.execute("CREATE INDEX idx_event_store_type ON core.event_store (event_type)")
    op.execute("CREATE INDEX idx_event_store_occurred ON core.event_store (occurred_at)")

    # Immutability: kein UPDATE/DELETE auf Event Store durch die allgemeine
    # Anwendungsrolle (BACKEND_ARCHITECTURE.md §2.1). Ohne dedizierte
    # Anwendungsrolle (folgt mit Auth-Schema in Schritt 2/2.7) betrifft das
    # aktuell nur PUBLIC — schadet nicht, greift aber erst vollständig, sobald
    # die App-Rolle nicht mehr der Tabellenbesitzer ist.
    op.execute("REVOKE UPDATE, DELETE ON core.event_store FROM PUBLIC")

    op.execute(
        """
        CREATE TABLE core.processed_commands (
            command_id      UUID PRIMARY KEY,
            aggregate_id    UUID NOT NULL,
            result_summary  JSONB NOT NULL,
            processed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS core.processed_commands")
    op.execute("DROP TABLE IF EXISTS core.event_store")
    op.execute("DROP SCHEMA IF EXISTS core CASCADE")
