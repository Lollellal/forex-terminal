"""Commands des TradeAllocation-Lifecycles. Siehe BACKEND_ARCHITECTURE.md
§2.2/§2.4. Reine Absichts-DTOs — Validierung/Invarianten leben im Aggregate
(trade_allocation.py), nicht hier."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from backend.domain.shared.command import Command


@dataclass(frozen=True, kw_only=True)
class CreateAllocationCommand(Command):
    account_id: uuid.UUID
    pair: str
    direction: str
    planned_risk_pct: Decimal
    signal_id: uuid.UUID | None = None
    entry_price_planned: Decimal | None = None
    sl_price: Decimal | None = None
    tp_price: Decimal | None = None
    signal_snapshot: dict | None = None


@dataclass(frozen=True, kw_only=True)
class ConfirmAllocationCommand(Command):
    allocation_id: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class MarkAllocationOpenedCommand(Command):
    allocation_id: uuid.UUID
    opened_at: datetime | None = None


@dataclass(frozen=True, kw_only=True)
class CloseAllocationCommand(Command):
    allocation_id: uuid.UUID
    close_reason: str
    realized_r: Decimal
    closed_at: datetime | None = None
