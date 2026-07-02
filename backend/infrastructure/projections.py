"""Account-Balance- und Empire-Overview-Projektion (Implementierungsschritt 2).

Projektionen sind eventually consistent und aus core.event_store jederzeit
neu aufbaubar (BACKEND_ARCHITECTURE.md §2.1) — hier per explizitem
catch_up()-Aufruf statt Background-Worker, da API/Scheduler nicht Teil
dieses Schritts sind. Checkpoint-Fortschritt liegt in
projections.checkpoints (global_seq je Projektion), damit ein wiederholter
catch_up()-Aufruf nur neue Events verarbeitet.
"""

from __future__ import annotations

import json

from sqlalchemy import Connection, text

from backend.domain.account import events as account_events
from backend.domain.allocation import events as allocation_events
from backend.domain.empire import events as empire_events
from backend.domain.journal import events as journal_events
from backend.domain.shared.event_envelope import EventEnvelope
from backend.domain.shared.event_store_protocol import EventStoreProtocol
from backend.domain.weekly_report import events as weekly_report_events

_GET_CHECKPOINT_SQL = text(
    "SELECT last_processed_seq FROM projections.checkpoints WHERE projection_name = :name"
)
_UPSERT_CHECKPOINT_SQL = text(
    """
    INSERT INTO projections.checkpoints (projection_name, last_processed_seq)
    VALUES (:name, :seq)
    ON CONFLICT (projection_name) DO UPDATE SET last_processed_seq = EXCLUDED.last_processed_seq
    """
)


def _get_checkpoint(conn: Connection, name: str) -> int:
    row = conn.execute(_GET_CHECKPOINT_SQL, {"name": name}).first()
    return row[0] if row else 0


def _set_checkpoint(conn: Connection, name: str, seq: int) -> None:
    conn.execute(_UPSERT_CHECKPOINT_SQL, {"name": name, "seq": seq})


class AccountBalanceProjection:
    """Ein Zeilenabbild pro Account, direkt aus AccountCreated/AccountUpdated."""

    name = "account_balances"

    _INSERT_SQL = text(
        """
        INSERT INTO projections.account_balances
            (account_id, empire_id, account_type, status, balance, equity, updated_at)
        VALUES
            (:account_id, :empire_id, :account_type, :status, :balance, :equity, now())
        ON CONFLICT (account_id) DO UPDATE SET
            empire_id    = EXCLUDED.empire_id,
            account_type = EXCLUDED.account_type,
            status       = EXCLUDED.status,
            balance      = EXCLUDED.balance,
            equity       = EXCLUDED.equity,
            updated_at   = now()
        """
    )
    _UPDATE_SQL = text(
        """
        UPDATE projections.account_balances
        SET balance = :balance, equity = :equity, status = :status, updated_at = now()
        WHERE account_id = :account_id
        """
    )

    def handles(self, event_type: str) -> bool:
        return event_type in (account_events.ACCOUNT_CREATED, account_events.ACCOUNT_UPDATED)

    def apply(self, conn: Connection, event: EventEnvelope) -> None:
        if event.event_type == account_events.ACCOUNT_CREATED:
            empire_id = event.payload["empire_id"]
            conn.execute(
                self._INSERT_SQL,
                {
                    "account_id": str(event.aggregate_id),
                    "empire_id": empire_id,
                    "account_type": event.payload["account_type"],
                    "status": "ACTIVE",
                    "balance": event.payload["balance"],
                    "equity": event.payload["equity"],
                },
            )
        elif event.event_type == account_events.ACCOUNT_UPDATED:
            conn.execute(
                self._UPDATE_SQL,
                {
                    "account_id": str(event.aggregate_id),
                    "balance": event.payload["balance"],
                    "equity": event.payload["equity"],
                    "status": event.payload["status"],
                },
            )


class EmpireOverviewProjection:
    """Ein Zeilenabbild pro Empire: Name aus EmpireCreated, Summen aus einem
    Recompute über projections.account_balances (muss daher nach der
    Account-Balance-Projektion catch-up'en, s. ProjectionRunner)."""

    name = "empire_overview"

    _UPSERT_NAME_SQL = text(
        """
        INSERT INTO projections.empire_overview (empire_id, name, updated_at)
        VALUES (:empire_id, :name, now())
        ON CONFLICT (empire_id) DO UPDATE SET name = EXCLUDED.name, updated_at = now()
        """
    )
    _RECOMPUTE_TOTALS_SQL = text(
        """
        INSERT INTO projections.empire_overview
            (empire_id, name, account_count, total_balance, total_equity, updated_at)
        SELECT
            :empire_id,
            COALESCE((SELECT name FROM projections.empire_overview WHERE empire_id = :empire_id), ''),
            COUNT(*),
            COALESCE(SUM(balance), 0),
            COALESCE(SUM(equity), 0),
            now()
        FROM projections.account_balances
        WHERE empire_id = :empire_id
        ON CONFLICT (empire_id) DO UPDATE SET
            account_count = EXCLUDED.account_count,
            total_balance = EXCLUDED.total_balance,
            total_equity  = EXCLUDED.total_equity,
            updated_at    = now()
        """
    )

    def handles(self, event_type: str) -> bool:
        return event_type in (
            empire_events.EMPIRE_CREATED,
            account_events.ACCOUNT_CREATED,
            account_events.ACCOUNT_UPDATED,
        )

    def apply(self, conn: Connection, event: EventEnvelope) -> None:
        if event.event_type == empire_events.EMPIRE_CREATED:
            conn.execute(
                self._UPSERT_NAME_SQL,
                {"empire_id": str(event.aggregate_id), "name": event.payload["name"]},
            )
            return

        empire_id = (
            event.payload["empire_id"]
            if event.event_type == account_events.ACCOUNT_CREATED
            else self._empire_id_of(conn, event.aggregate_id)
        )
        if empire_id is None:
            return
        conn.execute(self._RECOMPUTE_TOTALS_SQL, {"empire_id": empire_id})

    @staticmethod
    def _empire_id_of(conn: Connection, account_id) -> str | None:
        row = conn.execute(
            text("SELECT empire_id FROM projections.account_balances WHERE account_id = :aid"),
            {"aid": str(account_id)},
        ).first()
        return str(row[0]) if row and row[0] else None


class AllocationOverviewProjection:
    """Ein Zeilenabbild pro Allocation, aktiv und geschlossen gemeinsam —
    Filter über die status-Spalte (status <> 'CLOSED' = aktiv)."""

    name = "allocation_overview"

    _INSERT_SQL = text(
        """
        INSERT INTO projections.allocation_overview
            (allocation_id, account_id, pair, direction, status, planned_risk_pct, updated_at)
        VALUES
            (:allocation_id, :account_id, :pair, :direction, 'CREATED', :planned_risk_pct, now())
        ON CONFLICT (allocation_id) DO UPDATE SET
            account_id       = EXCLUDED.account_id,
            pair             = EXCLUDED.pair,
            direction        = EXCLUDED.direction,
            status           = EXCLUDED.status,
            planned_risk_pct = EXCLUDED.planned_risk_pct,
            updated_at       = now()
        """
    )
    _CONFIRMED_SQL = text(
        """
        UPDATE projections.allocation_overview
        SET status = 'CONFIRMED', applied_risk_pct = :applied_risk_pct, updated_at = now()
        WHERE allocation_id = :allocation_id
        """
    )
    _OPENED_SQL = text(
        """
        UPDATE projections.allocation_overview
        SET status = 'OPEN', applied_risk_pct = :applied_risk_pct, updated_at = now()
        WHERE allocation_id = :allocation_id
        """
    )
    _CLOSED_SQL = text(
        """
        UPDATE projections.allocation_overview
        SET status = 'CLOSED', closed_at = :closed_at, close_reason = :close_reason,
            realized_r = :realized_r, updated_at = now()
        WHERE allocation_id = :allocation_id
        """
    )

    def handles(self, event_type: str) -> bool:
        return event_type in (
            allocation_events.ALLOCATION_CREATED,
            allocation_events.ALLOCATION_CONFIRMED,
            allocation_events.ALLOCATION_OPENED,
            allocation_events.ALLOCATION_CLOSED,
        )

    def apply(self, conn: Connection, event: EventEnvelope) -> None:
        allocation_id = str(event.aggregate_id)
        if event.event_type == allocation_events.ALLOCATION_CREATED:
            conn.execute(
                self._INSERT_SQL,
                {
                    "allocation_id": allocation_id,
                    "account_id": event.payload["account_id"],
                    "pair": event.payload["pair"],
                    "direction": event.payload["direction"],
                    "planned_risk_pct": event.payload["planned_risk_pct"],
                },
            )
        elif event.event_type == allocation_events.ALLOCATION_CONFIRMED:
            conn.execute(
                self._CONFIRMED_SQL,
                {
                    "allocation_id": allocation_id,
                    "applied_risk_pct": event.payload["applied_risk_pct"],
                },
            )
        elif event.event_type == allocation_events.ALLOCATION_OPENED:
            conn.execute(
                self._OPENED_SQL,
                {
                    "allocation_id": allocation_id,
                    "applied_risk_pct": event.payload["applied_risk_pct"],
                },
            )
        elif event.event_type == allocation_events.ALLOCATION_CLOSED:
            conn.execute(
                self._CLOSED_SQL,
                {
                    "allocation_id": allocation_id,
                    "closed_at": event.payload["closed_at"],
                    "close_reason": event.payload["close_reason"],
                    "realized_r": event.payload["realized_r"],
                },
            )


class JournalProjection:
    """Eine Zeile pro Allocation, kombiniert Allocation-Lifecycle, einen
    Account-Snapshot bei AllocationCreated (keine Live-Ansicht — spätere
    AccountUpdated-Events werden nicht konsumiert, Journal ist Historie) und
    eingebettete JournalNotes. Journal bleibt reine Projektion, keine eigene
    Wahrheit — JournalNote-Events tragen selbst keine Trade-Fakten, die
    Trade-Fakten kommen ausschließlich aus den Allocation-Events."""

    name = "journal_view"

    _ACCOUNT_SNAPSHOT_SQL = text(
        "SELECT account_type, balance, equity FROM core.accounts WHERE id = :account_id"
    )
    _INSERT_SQL = text(
        """
        INSERT INTO projections.journal_view
            (allocation_id, account_id, pair, direction, status, planned_risk_pct,
             account_snapshot, notes, updated_at)
        VALUES
            (:allocation_id, :account_id, :pair, :direction, 'CREATED', :planned_risk_pct,
             CAST(:account_snapshot AS jsonb), '[]', now())
        ON CONFLICT (allocation_id) DO UPDATE SET
            account_id       = EXCLUDED.account_id,
            pair             = EXCLUDED.pair,
            direction        = EXCLUDED.direction,
            status           = EXCLUDED.status,
            planned_risk_pct = EXCLUDED.planned_risk_pct,
            account_snapshot = EXCLUDED.account_snapshot,
            updated_at       = now()
        """
    )
    _CONFIRMED_SQL = text(
        """
        UPDATE projections.journal_view
        SET status = 'CONFIRMED', applied_risk_pct = :applied_risk_pct, updated_at = now()
        WHERE allocation_id = :allocation_id
        """
    )
    _OPENED_SQL = text(
        """
        UPDATE projections.journal_view
        SET status = 'OPEN', applied_risk_pct = :applied_risk_pct, updated_at = now()
        WHERE allocation_id = :allocation_id
        """
    )
    _CLOSED_SQL = text(
        """
        UPDATE projections.journal_view
        SET status = 'CLOSED', closed_at = :closed_at, close_reason = :close_reason,
            realized_r = :realized_r, updated_at = now()
        WHERE allocation_id = :allocation_id
        """
    )
    _SELECT_NOTES_SQL = text(
        "SELECT notes FROM projections.journal_view WHERE allocation_id = :allocation_id"
    )
    _UPDATE_NOTES_SQL = text(
        """
        UPDATE projections.journal_view
        SET notes = CAST(:notes AS jsonb), updated_at = now()
        WHERE allocation_id = :allocation_id
        """
    )

    def handles(self, event_type: str) -> bool:
        return event_type in (
            allocation_events.ALLOCATION_CREATED,
            allocation_events.ALLOCATION_CONFIRMED,
            allocation_events.ALLOCATION_OPENED,
            allocation_events.ALLOCATION_CLOSED,
            journal_events.JOURNAL_NOTE_ADDED,
            journal_events.JOURNAL_NOTE_EDITED,
        )

    def apply(self, conn: Connection, event: EventEnvelope) -> None:
        if event.event_type == allocation_events.ALLOCATION_CREATED:
            self._apply_created(conn, event)
        elif event.event_type == allocation_events.ALLOCATION_CONFIRMED:
            conn.execute(
                self._CONFIRMED_SQL,
                {
                    "allocation_id": str(event.aggregate_id),
                    "applied_risk_pct": event.payload["applied_risk_pct"],
                },
            )
        elif event.event_type == allocation_events.ALLOCATION_OPENED:
            conn.execute(
                self._OPENED_SQL,
                {
                    "allocation_id": str(event.aggregate_id),
                    "applied_risk_pct": event.payload["applied_risk_pct"],
                },
            )
        elif event.event_type == allocation_events.ALLOCATION_CLOSED:
            conn.execute(
                self._CLOSED_SQL,
                {
                    "allocation_id": str(event.aggregate_id),
                    "closed_at": event.payload["closed_at"],
                    "close_reason": event.payload["close_reason"],
                    "realized_r": event.payload["realized_r"],
                },
            )
        elif event.event_type == journal_events.JOURNAL_NOTE_ADDED:
            self._apply_note_added(conn, event)
        elif event.event_type == journal_events.JOURNAL_NOTE_EDITED:
            self._apply_note_edited(conn, event)

    def _apply_created(self, conn: Connection, event: EventEnvelope) -> None:
        account_id = event.payload["account_id"]
        account_row = conn.execute(self._ACCOUNT_SNAPSHOT_SQL, {"account_id": account_id}).one()
        snapshot = {
            "account_type": account_row.account_type,
            "balance": str(account_row.balance),
            "equity": str(account_row.equity),
        }
        conn.execute(
            self._INSERT_SQL,
            {
                "allocation_id": str(event.aggregate_id),
                "account_id": account_id,
                "pair": event.payload["pair"],
                "direction": event.payload["direction"],
                "planned_risk_pct": event.payload["planned_risk_pct"],
                "account_snapshot": json.dumps(snapshot),
            },
        )

    def _apply_note_added(self, conn: Connection, event: EventEnvelope) -> None:
        allocation_id = event.payload["related_allocation_id"]
        if not allocation_id:
            return  # Notiz an einem Signal, nicht an einer Allocation — außerhalb des Journal-View-Scopes
        row = conn.execute(self._SELECT_NOTES_SQL, {"allocation_id": allocation_id}).first()
        if row is None:
            return  # Allocation-Zeile noch nicht materialisiert (Checkpoint-Reihenfolge), defensiv
        notes = list(row[0] or [])
        notes.append(
            {
                "note_id": str(event.aggregate_id),
                "text": event.payload["text"],
                "attachments": event.payload["attachments"],
                "created_at": event.occurred_at.isoformat(),
            }
        )
        conn.execute(
            self._UPDATE_NOTES_SQL, {"allocation_id": allocation_id, "notes": json.dumps(notes)}
        )

    def _apply_note_edited(self, conn: Connection, event: EventEnvelope) -> None:
        note_id = str(event.aggregate_id)
        rows = conn.execute(
            text(
                "SELECT allocation_id, notes FROM projections.journal_view "
                "WHERE notes @> :probe"
            ),
            {"probe": json.dumps([{"note_id": note_id}])},
        ).all()
        for allocation_id, notes in rows:
            updated = list(notes)
            for entry in updated:
                if entry.get("note_id") == note_id:
                    entry["text"] = event.payload["text"]
                    entry["attachments"] = event.payload["attachments"]
                    entry["edited_at"] = event.occurred_at.isoformat()
            conn.execute(
                self._UPDATE_NOTES_SQL, {"allocation_id": allocation_id, "notes": json.dumps(updated)}
            )


class WeeklyReportProjection:
    """Ein Zeilenabbild pro WeeklyReport — Mobile liest aus projections,
    nie aus core.weekly_reports direkt (BACKEND_ARCHITECTURE.md §2.5/§2.6)."""

    name = "weekly_reports"

    _INSERT_SQL = text(
        """
        INSERT INTO projections.weekly_reports
            (id, user_id, period_start, period_end, status, content_ref, updated_at)
        VALUES
            (:id, :user_id, :period_start, :period_end, 'GENERATED', :content_ref, now())
        ON CONFLICT (id) DO UPDATE SET
            user_id      = EXCLUDED.user_id,
            period_start = EXCLUDED.period_start,
            period_end   = EXCLUDED.period_end,
            status       = EXCLUDED.status,
            content_ref  = EXCLUDED.content_ref,
            updated_at   = now()
        """
    )
    _PUBLISHED_SQL = text(
        """
        UPDATE projections.weekly_reports
        SET status = 'PUBLISHED', published_at = :published_at, updated_at = now()
        WHERE id = :id
        """
    )

    def handles(self, event_type: str) -> bool:
        return event_type in (
            weekly_report_events.WEEKLY_REPORT_GENERATED,
            weekly_report_events.WEEKLY_REPORT_PUBLISHED,
        )

    def apply(self, conn: Connection, event: EventEnvelope) -> None:
        if event.event_type == weekly_report_events.WEEKLY_REPORT_GENERATED:
            conn.execute(
                self._INSERT_SQL,
                {
                    "id": str(event.aggregate_id),
                    "user_id": event.payload["user_id"],
                    "period_start": event.payload["period_start"],
                    "period_end": event.payload["period_end"],
                    "content_ref": event.payload["content_ref"],
                },
            )
        elif event.event_type == weekly_report_events.WEEKLY_REPORT_PUBLISHED:
            conn.execute(
                self._PUBLISHED_SQL,
                {"id": str(event.aggregate_id), "published_at": event.payload["published_at"]},
            )


class ProjectionRunner:
    """Fährt beide Projektionen gegen core.event_store nach — in fester
    Reihenfolge (Account-Balances vor Empire-Overview), damit der
    Totals-Recompute der Empire-Projektion auf bereits aktuellen
    Account-Balance-Zeilen aufsetzt."""

    def __init__(self, event_store: EventStoreProtocol) -> None:
        self._event_store = event_store
        self._projections = (
            AccountBalanceProjection(),
            EmpireOverviewProjection(),
            AllocationOverviewProjection(),
            JournalProjection(),
            WeeklyReportProjection(),
        )

    def catch_up(self, conn: Connection, batch_size: int = 1000) -> None:
        for projection in self._projections:
            self._catch_up_one(conn, projection, batch_size)

    def _catch_up_one(self, conn: Connection, projection, batch_size: int) -> None:
        after_seq = _get_checkpoint(conn, projection.name)
        while True:
            batch = self._event_store.load_all(conn, after_seq=after_seq, limit=batch_size)
            if not batch:
                break
            for seq, event in batch:
                if projection.handles(event.event_type):
                    projection.apply(conn, event)
                after_seq = seq
            _set_checkpoint(conn, projection.name, after_seq)
            if len(batch) < batch_size:
                break
