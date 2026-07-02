"""Market-Snapshot-Endpunkt (v1.1 Mobile Trade Intelligence). Rein lesend —
der Wert wird nicht über die API geschrieben, sondern direkt vom
Desktop-Publish-Skript in projections.market_snapshot upgesetzt (kein
Aggregate, keine Business-Regeln, siehe Migration 0015)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Connection, text

from backend.api.dependencies import get_read_conn
from backend.api.schemas.market_snapshot import MarketSnapshotResponse

router = APIRouter(prefix="/market-snapshot", tags=["market-snapshot"])

_SELECT_SQL = text(
    "SELECT regime, vix, yield_curve, updated_at FROM projections.market_snapshot WHERE id = 'current'"
)


@router.get("", response_model=MarketSnapshotResponse)
def get_market_snapshot(conn: Connection = Depends(get_read_conn)):
    row = conn.execute(_SELECT_SQL).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Noch kein Market-Snapshot veröffentlicht")
    return MarketSnapshotResponse(**row)
