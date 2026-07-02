"""Pydantic-Contracts für den WeeklyReport-Endpunkt."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel


class WeeklyReportResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    period_start: date
    period_end: date
    status: str
    content_ref: str | None
    summary: str | None = None
    published_at: datetime | None


class WeeklyReportDownloadUrlResponse(BaseModel):
    url: str
    expires_in: int
