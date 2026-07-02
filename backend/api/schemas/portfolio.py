"""Pydantic-Contracts für das Home-Dashboard (Implementierungsschritt 7,
MOBILE_DATA_CONTRACT.md Abschnitt "Home"). Reine Aggregation bestehender
Projektionen, kein neues Domain-Konzept."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class EmpireSummary(BaseModel):
    empire_id: uuid.UUID
    name: str
    account_count: int
    total_balance: Decimal
    total_equity: Decimal


class StandaloneAccountSummary(BaseModel):
    account_id: uuid.UUID
    account_type: str
    status: str
    balance: Decimal
    equity: Decimal


class RecentJournalEntry(BaseModel):
    allocation_id: uuid.UUID
    pair: str
    status: str
    updated_at: datetime


class PortfolioResponse(BaseModel):
    user_id: uuid.UUID
    total_balance: Decimal
    total_equity: Decimal
    empires: list[EmpireSummary]
    standalone_accounts: list[StandaloneAccountSummary]
    active_trade_count: int
    recent_journal_entries: list[RecentJournalEntry]
