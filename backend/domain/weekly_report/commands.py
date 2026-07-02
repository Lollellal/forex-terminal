"""Commands des WeeklyReport-Layers. Siehe BACKEND_ARCHITECTURE.md §2.2.
RegisterWeeklyReportCommand trägt bewusst die PDF-Bytes — der Command ist
der einzige Ort, an dem Upload + Registrierung zusammenkommen (siehe
weekly_report_service.py)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from backend.domain.shared.command import Command


@dataclass(frozen=True, kw_only=True)
class RegisterWeeklyReportCommand(Command):
    user_id: uuid.UUID
    period_start: date
    period_end: date
    content: bytes
    content_type: str = "application/pdf"


@dataclass(frozen=True, kw_only=True)
class PublishWeeklyReportCommand(Command):
    report_id: uuid.UUID
