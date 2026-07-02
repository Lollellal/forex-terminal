"""Pydantic-Contract für den Market-Snapshot-Endpunkt (v1.1 Mobile Trade Intelligence)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class MarketSnapshotResponse(BaseModel):
    regime: str
    vix: Decimal | None
    yield_curve: Decimal | None
    updated_at: datetime
