"""Allocation-Endpunkte. Reine Übersetzung Request -> Command ->
AllocationLifecycleService -> Response. REJECT-Entscheidungen des Risk Gate
werden nicht hier behandelt, sondern über den globalen
RiskGateRejectedError-Handler in app.py (409) — der Router muss den Risk
Gate nicht kennen, nur den Service aufrufen."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Connection, text

from backend.api.dependencies import get_read_conn, get_write_conn
from backend.api.schemas.allocation import (
    AllocationOverviewResponse,
    AllocationResponse,
    CloseAllocationRequest,
    CreateAllocationRequest,
    MarkAllocationOpenedRequest,
)
from backend.domain.account.account_repository import AccountRepository
from backend.domain.account.account_service import AccountService
from backend.domain.allocation.allocation_lifecycle_service import AllocationLifecycleService
from backend.domain.allocation.allocation_repository import AllocationRepository
from backend.domain.allocation.commands import (
    CloseAllocationCommand,
    ConfirmAllocationCommand,
    CreateAllocationCommand,
    MarkAllocationOpenedCommand,
)
from backend.domain.risk.policies.consecutive_losses_policy import ConsecutiveLossesPolicy
from backend.domain.risk.policies.same_pair_policy import SamePairOpenPolicy
from backend.domain.risk.risk_gate_service import RiskGateService
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner

router = APIRouter(prefix="/allocations", tags=["allocations"])

_service = AllocationLifecycleService(
    AllocationRepository(EventStore()),
    RiskGateService([SamePairOpenPolicy(), ConsecutiveLossesPolicy()]),
)
_account_service = AccountService(AccountRepository(EventStore()))
_projections = ProjectionRunner(EventStore())


@router.post("", response_model=AllocationResponse, status_code=201)
def create_allocation(payload: CreateAllocationRequest, conn: Connection = Depends(get_write_conn)):
    allocation = _service.create(
        conn,
        CreateAllocationCommand(
            account_id=payload.account_id,
            pair=payload.pair,
            direction=payload.direction,
            planned_risk_pct=payload.planned_risk_pct,
            signal_id=payload.signal_id,
            entry_price_planned=payload.entry_price_planned,
            sl_price=payload.sl_price,
            tp_price=payload.tp_price,
            signal_snapshot=payload.signal_snapshot,
        ),
    )
    _projections.catch_up(conn)
    return _to_response(allocation)


@router.post("/{allocation_id}/confirm", response_model=AllocationResponse)
def confirm_allocation(allocation_id: uuid.UUID, conn: Connection = Depends(get_write_conn)):
    allocation = _service.confirm(conn, ConfirmAllocationCommand(allocation_id=allocation_id))
    _projections.catch_up(conn)
    return _to_response(allocation)


@router.post("/{allocation_id}/open", response_model=AllocationResponse)
def mark_allocation_opened(
    allocation_id: uuid.UUID, payload: MarkAllocationOpenedRequest, conn: Connection = Depends(get_write_conn)
):
    allocation = _service.mark_opened(
        conn,
        MarkAllocationOpenedCommand(allocation_id=allocation_id, opened_at=payload.opened_at),
    )
    _projections.catch_up(conn)
    return _to_response(allocation)


@router.post("/{allocation_id}/close", response_model=AllocationResponse)
def close_allocation(
    allocation_id: uuid.UUID, payload: CloseAllocationRequest, conn: Connection = Depends(get_write_conn)
):
    allocation = _service.close(
        conn,
        CloseAllocationCommand(
            allocation_id=allocation_id,
            close_reason=payload.close_reason,
            realized_r=payload.realized_r,
            closed_at=payload.closed_at,
        ),
    )
    _account_service.recompute_from_closed_allocations(conn, allocation.account_id)
    _projections.catch_up(conn)
    return _to_response(allocation)


@router.get("", response_model=list[AllocationOverviewResponse])
def list_allocations(
    account_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    status: str | None = None,
    conn: Connection = Depends(get_read_conn),
):
    """``user_id`` joint über core.accounts, damit Mobile "alle aktiven
    Trades des Users" ohne N+1 über jeden Account einzeln abfragen kann
    (MOBILE_DATA_CONTRACT.md, Abschnitt "Active Trades"). ``status=ACTIVE``
    ist ein Alias für CONFIRMED+OPEN — reine Query-Bequemlichkeit, keine
    neue Domain-Regel."""
    query = "SELECT ao.* FROM projections.allocation_overview ao WHERE 1=1"
    params: dict[str, object] = {}
    if account_id is not None:
        query += " AND ao.account_id = :account_id"
        params["account_id"] = str(account_id)
    if user_id is not None:
        query += " AND ao.account_id IN (SELECT id FROM core.accounts WHERE user_id = :user_id)"
        params["user_id"] = str(user_id)
    if status is not None:
        if status == "ACTIVE":
            query += " AND ao.status IN ('CONFIRMED','OPEN')"
        else:
            query += " AND ao.status = :status"
            params["status"] = status
    query += " ORDER BY ao.updated_at DESC"
    rows = conn.execute(text(query), params).mappings().all()
    return [AllocationOverviewResponse(**row) for row in rows]


@router.get("/{allocation_id}", response_model=AllocationOverviewResponse)
def get_allocation(allocation_id: uuid.UUID, conn: Connection = Depends(get_read_conn)):
    row = conn.execute(
        text("SELECT * FROM projections.allocation_overview WHERE allocation_id = :id"),
        {"id": str(allocation_id)},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Allocation {allocation_id} nicht in der Projektion gefunden")
    return AllocationOverviewResponse(**row)


def _to_response(allocation) -> AllocationResponse:
    return AllocationResponse(
        id=allocation.id,
        account_id=allocation.account_id,
        pair=allocation.pair,
        direction=allocation.direction,
        status=allocation.status,
        planned_risk_pct=allocation.planned_risk_pct,
        applied_risk_pct=allocation.applied_risk_pct,
        close_reason=allocation.close_reason,
        realized_r=allocation.realized_r,
        signal_snapshot=allocation.signal_snapshot,
        version=allocation.version,
    )
