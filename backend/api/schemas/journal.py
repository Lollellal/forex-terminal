"""Pydantic-Contracts für den Journal-Endpunkt."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel


class AddJournalNoteRequest(BaseModel):
    text: str
    related_allocation_id: uuid.UUID | None = None
    related_signal_id: uuid.UUID | None = None
    attachments: list[str] = []


class EditJournalNoteRequest(BaseModel):
    text: str
    attachments: list[str] | None = None


class JournalNoteResponse(BaseModel):
    id: uuid.UUID
    related_allocation_id: uuid.UUID | None
    related_signal_id: uuid.UUID | None
    text: str
    attachments: list[str]
    version: int


class JournalViewResponse(BaseModel):
    allocation_id: uuid.UUID
    account_id: uuid.UUID
    pair: str
    direction: str
    status: str
    planned_risk_pct: Decimal
    applied_risk_pct: Decimal | None
    closed_at: datetime | None
    close_reason: str | None
    realized_r: Decimal | None
    account_snapshot: dict[str, Any]
    notes: list[dict[str, Any]]
