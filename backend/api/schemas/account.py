"""Pydantic-Contracts für den Account-Endpunkt. Reine Übersetzung von/zu den
bestehenden Domain-Commands/Aggregaten — keine eigene Validierungslogik
über das Aggregat hinaus."""

from __future__ import annotations

import uuid
from decimal import Decimal

from pydantic import BaseModel


class CreateAccountRequest(BaseModel):
    user_id: uuid.UUID
    account_type: str
    initial_balance: Decimal
    initial_equity: Decimal
    empire_id: uuid.UUID | None = None


class UpdateAccountBalanceRequest(BaseModel):
    balance: Decimal
    equity: Decimal


class AccountResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    empire_id: uuid.UUID | None
    account_type: str
    status: str
    balance: Decimal
    equity: Decimal
    version: int


class AccountBalanceProjectionResponse(BaseModel):
    account_id: uuid.UUID
    empire_id: uuid.UUID | None
    account_type: str
    status: str
    balance: Decimal
    equity: Decimal
