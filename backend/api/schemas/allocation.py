"""Pydantic-Contracts für den Allocation-Endpunkt."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class CreateAllocationRequest(BaseModel):
    account_id: uuid.UUID
    pair: str
    direction: str
    planned_risk_pct: Decimal
    signal_id: uuid.UUID | None = None
    entry_price_planned: Decimal | None = None
    sl_price: Decimal | None = None
    tp_price: Decimal | None = None


class MarkAllocationOpenedRequest(BaseModel):
    opened_at: datetime | None = None


class CloseAllocationRequest(BaseModel):
    close_reason: str
    realized_r: Decimal
    closed_at: datetime | None = None


class AllocationResponse(BaseModel):
    id: uuid.UUID
    account_id: uuid.UUID
    pair: str
    direction: str
    status: str
    planned_risk_pct: Decimal
    applied_risk_pct: Decimal | None
    close_reason: str | None
    realized_r: Decimal | None
    version: int


class AllocationOverviewResponse(BaseModel):
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


class RiskGateRejectedResponse(BaseModel):
    detail: str
    triggered_policy: str | None
