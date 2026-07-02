"""core.journal_notes (Implementierungsschritt 5)

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-02

JournalNote trägt bewusst KEINE Spalten für Ergebnis/R-Multiple/Preise/
Status (BACKEND_ARCHITECTURE.md §2.1) — "JournalNote speichert niemals
Trade-Fakten" ist damit nicht nur Konvention, sondern durch fehlende Spalten
strukturell unmöglich. related_signal_id ist ohne FK (core.signals existiert
noch nicht, Signal-Aggregate ist nicht Teil dieses Schritts) — nur ein
Platzhalter, damit der chk_exactly_one_reference-Constraint schon das
Zielschema abbildet.
"""

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE core.journal_notes (
            id                     UUID PRIMARY KEY,
            related_allocation_id  UUID NULL REFERENCES core.trade_allocations(id),
            related_signal_id      UUID NULL,
            text                   TEXT NOT NULL,
            attachments            JSONB NOT NULL DEFAULT '[]',
            version                INTEGER NOT NULL DEFAULT 0,
            created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
            edited_at              TIMESTAMPTZ NULL,
            CONSTRAINT chk_exactly_one_reference CHECK (
                (related_allocation_id IS NOT NULL)::int + (related_signal_id IS NOT NULL)::int = 1
            )
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_journal_notes_allocation ON core.journal_notes (related_allocation_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS core.journal_notes")
