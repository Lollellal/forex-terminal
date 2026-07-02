"""Account-Endpunkte. Reine Übersetzung Request -> Command -> Service ->
Response — keine Business-Logik hier, die liegt vollständig in
backend.domain.account."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Connection, text

from backend.api.dependencies import get_read_conn, get_write_conn
from backend.api.schemas.account import (
    AccountBalanceProjectionResponse,
    AccountResponse,
    CreateAccountRequest,
    UpdateAccountBalanceRequest,
)
from backend.domain.account.account_repository import AccountRepository
from backend.domain.account.account_service import AccountService
from backend.domain.account.commands import CreateAccountCommand, UpdateAccountBalanceCommand
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner

router = APIRouter(prefix="/accounts", tags=["accounts"])

_service = AccountService(AccountRepository(EventStore()))
_projections = ProjectionRunner(EventStore())


@router.post("", response_model=AccountResponse, status_code=201)
def create_account(payload: CreateAccountRequest, conn: Connection = Depends(get_write_conn)):
    account = _service.create(
        conn,
        CreateAccountCommand(
            user_id=payload.user_id,
            account_type=payload.account_type,
            initial_balance=payload.initial_balance,
            initial_equity=payload.initial_equity,
            empire_id=payload.empire_id,
        ),
    )
    _projections.catch_up(conn)
    return _to_response(account)


@router.post("/{account_id}/balance", response_model=AccountResponse)
def update_balance(
    account_id: uuid.UUID, payload: UpdateAccountBalanceRequest, conn: Connection = Depends(get_write_conn)
):
    account = _service.update_balance(
        conn,
        UpdateAccountBalanceCommand(account_id=account_id, balance=payload.balance, equity=payload.equity),
    )
    _projections.catch_up(conn)
    return _to_response(account)


@router.get("/{account_id}/balance", response_model=AccountBalanceProjectionResponse)
def get_account_balance(account_id: uuid.UUID, conn: Connection = Depends(get_read_conn)):
    row = conn.execute(
        text(
            "SELECT account_id, empire_id, account_type, status, balance, equity "
            "FROM projections.account_balances WHERE account_id = :id"
        ),
        {"id": str(account_id)},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Account {account_id} nicht in der Projektion gefunden")
    return AccountBalanceProjectionResponse(**row)


def _to_response(account) -> AccountResponse:
    return AccountResponse(
        id=account.id,
        user_id=account.user_id,
        empire_id=account.empire_id,
        account_type=account.account_type,
        status=account.status,
        balance=account.balance,
        equity=account.equity,
        version=account.version,
    )
