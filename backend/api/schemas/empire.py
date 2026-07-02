"""Pydantic-Contracts für den Empire-Endpunkt."""

from __future__ import annotations

import uuid
from decimal import Decimal

from pydantic import BaseModel


class CreateEmpireRequest(BaseModel):
    user_id: uuid.UUID
    name: str


class EmpireResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    version: int


class EmpireOverviewResponse(BaseModel):
    empire_id: uuid.UUID
    name: str
    account_count: int
    total_balance: Decimal
    total_equity: Decimal
