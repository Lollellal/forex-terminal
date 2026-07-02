"""Commands des Account-Aggregates. Siehe BACKEND_ARCHITECTURE.md §2.2.
Reine Absichts-DTOs — Validierung/Invarianten leben im Aggregate
(account.py), nicht hier."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from backend.domain.shared.command import Command


@dataclass(frozen=True, kw_only=True)
class CreateAccountCommand(Command):
    user_id: uuid.UUID
    account_type: str
    initial_balance: Decimal
    initial_equity: Decimal
    empire_id: uuid.UUID | None = None


@dataclass(frozen=True, kw_only=True)
class UpdateAccountBalanceCommand(Command):
    account_id: uuid.UUID
    balance: Decimal
    equity: Decimal
