"""Empire-Endpunkte. Reine Übersetzung, keine Business-Logik."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Connection, text

from backend.api.dependencies import get_read_conn, get_write_conn
from backend.api.schemas.account import AccountBalanceProjectionResponse
from backend.api.schemas.empire import CreateEmpireRequest, EmpireOverviewResponse, EmpireResponse
from backend.domain.empire.commands import CreateEmpireCommand
from backend.domain.empire.empire_repository import EmpireRepository
from backend.domain.empire.empire_service import EmpireService
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner

router = APIRouter(prefix="/empires", tags=["empires"])

_service = EmpireService(EmpireRepository(EventStore()))
_projections = ProjectionRunner(EventStore())


@router.post("", response_model=EmpireResponse, status_code=201)
def create_empire(payload: CreateEmpireRequest, conn: Connection = Depends(get_write_conn)):
    empire = _service.create(conn, CreateEmpireCommand(user_id=payload.user_id, name=payload.name))
    _projections.catch_up(conn)
    return EmpireResponse(id=empire.id, user_id=empire.user_id, name=empire.name, version=empire.version)


@router.get("/{empire_id}/overview", response_model=EmpireOverviewResponse)
def get_empire_overview(empire_id: uuid.UUID, conn: Connection = Depends(get_read_conn)):
    row = conn.execute(
        text(
            "SELECT empire_id, name, account_count, total_balance, total_equity "
            "FROM projections.empire_overview WHERE empire_id = :id"
        ),
        {"id": str(empire_id)},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Empire {empire_id} nicht in der Projektion gefunden")
    return EmpireOverviewResponse(**row)


@router.get("/{empire_id}/accounts", response_model=list[AccountBalanceProjectionResponse])
def list_empire_accounts(empire_id: uuid.UUID, conn: Connection = Depends(get_read_conn)):
    """MOBILE_DATA_CONTRACT.md, Abschnitt "Empire Overview": welche Accounts
    gehören zu dieser Empire — fehlte bisher, empire_overview liefert nur
    Summen, keine Mitgliederliste."""
    rows = conn.execute(
        text(
            "SELECT account_id, empire_id, account_type, status, balance, equity "
            "FROM projections.account_balances WHERE empire_id = :id ORDER BY balance DESC"
        ),
        {"id": str(empire_id)},
    ).mappings().all()
    return [AccountBalanceProjectionResponse(**row) for row in rows]
