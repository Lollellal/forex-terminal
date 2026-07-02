"""Orchestriert Storage-Upload + Aggregat-Registrierung. Der Storage-Upload
läuft außerhalb der DB-Transaktion (Supabase Storage ist ein separates
System, keine verteilte Transaktion mit Postgres) — schlägt der
anschließende DB-Save fehl, bleibt eine verwaiste Datei im Bucket zurück.
Für den aktuellen Einsatzzweck (Desktop registriert nach erfolgreicher
Generierung, kein Hochfrequenz-Pfad) ist das ein akzeptierter Kompromiss,
kein automatisches Aufräumen in diesem Schritt."""

from __future__ import annotations

import uuid

from sqlalchemy import Connection

from backend.infrastructure.storage import WeeklyReportStorage

from .commands import PublishWeeklyReportCommand, RegisterWeeklyReportCommand
from .weekly_report import WeeklyReport
from .weekly_report_repository import WeeklyReportRepository


class WeeklyReportService:
    def __init__(self, repository: WeeklyReportRepository, storage: WeeklyReportStorage) -> None:
        self._repository = repository
        self._storage = storage

    def register(self, conn: Connection, command: RegisterWeeklyReportCommand) -> WeeklyReport:
        storage_path = (
            f"{command.user_id}/{command.period_start.isoformat()}_{command.period_end.isoformat()}.pdf"
        )
        content_ref = self._storage.upload(
            storage_path=storage_path, content=command.content, content_type=command.content_type
        )
        report = WeeklyReport.generate(
            uuid.uuid4(),
            user_id=command.user_id,
            period_start=command.period_start,
            period_end=command.period_end,
            content_ref=content_ref,
            summary=command.summary,
            source=command.source,
            correlation_id=command.correlation_id,
        )
        self._repository.save(conn, report)
        return report

    def publish(self, conn: Connection, command: PublishWeeklyReportCommand) -> WeeklyReport:
        report = self._repository.load(conn, command.report_id)
        report.publish(source=command.source, correlation_id=command.correlation_id)
        self._repository.save(conn, report)
        return report
