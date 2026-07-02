"""Journal-Endpunkte. Reine Übersetzung, keine Business-Logik."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Connection, text

from backend.api.dependencies import get_read_conn, get_write_conn
from backend.api.schemas.journal import (
    AddJournalNoteRequest,
    EditJournalNoteRequest,
    JournalNoteResponse,
    JournalViewResponse,
)
from backend.domain.journal.commands import AddJournalNoteCommand, EditJournalNoteCommand
from backend.domain.journal.journal_note_repository import JournalNoteRepository
from backend.domain.journal.journal_note_service import JournalNoteService
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner

router = APIRouter(tags=["journal"])

_service = JournalNoteService(JournalNoteRepository(EventStore()))
_projections = ProjectionRunner(EventStore())


@router.post("/journal-notes", response_model=JournalNoteResponse, status_code=201)
def add_journal_note(payload: AddJournalNoteRequest, conn: Connection = Depends(get_write_conn)):
    note = _service.add(
        conn,
        AddJournalNoteCommand(
            text=payload.text,
            related_allocation_id=payload.related_allocation_id,
            related_signal_id=payload.related_signal_id,
            attachments=payload.attachments,
        ),
    )
    _projections.catch_up(conn)
    return _to_response(note)


@router.patch("/journal-notes/{note_id}", response_model=JournalNoteResponse)
def edit_journal_note(
    note_id: uuid.UUID, payload: EditJournalNoteRequest, conn: Connection = Depends(get_write_conn)
):
    note = _service.edit(
        conn, EditJournalNoteCommand(note_id=note_id, text=payload.text, attachments=payload.attachments)
    )
    _projections.catch_up(conn)
    return _to_response(note)


@router.get("/journal", response_model=list[JournalViewResponse])
def list_journal(
    account_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    status: str | None = None,
    conn: Connection = Depends(get_read_conn),
):
    """``user_id`` joint über core.accounts, analog zu GET /allocations
    (MOBILE_DATA_CONTRACT.md, Abschnitt "Journal Update")."""
    query = "SELECT jv.* FROM projections.journal_view jv WHERE 1=1"
    params: dict[str, object] = {}
    if account_id is not None:
        query += " AND jv.account_id = :account_id"
        params["account_id"] = str(account_id)
    if user_id is not None:
        query += " AND jv.account_id IN (SELECT id FROM core.accounts WHERE user_id = :user_id)"
        params["user_id"] = str(user_id)
    if status is not None:
        query += " AND jv.status = :status"
        params["status"] = status
    query += " ORDER BY jv.updated_at DESC"
    rows = conn.execute(text(query), params).mappings().all()
    return [JournalViewResponse(**row) for row in rows]


@router.get("/journal/{allocation_id}", response_model=JournalViewResponse)
def get_journal_entry(allocation_id: uuid.UUID, conn: Connection = Depends(get_read_conn)):
    row = conn.execute(
        text("SELECT * FROM projections.journal_view WHERE allocation_id = :id"),
        {"id": str(allocation_id)},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Journal-Eintrag für Allocation {allocation_id} nicht gefunden")
    return JournalViewResponse(**row)


def _to_response(note) -> JournalNoteResponse:
    return JournalNoteResponse(
        id=note.id,
        related_allocation_id=note.related_allocation_id,
        related_signal_id=note.related_signal_id,
        text=note.text,
        attachments=note.attachments,
        version=note.version,
    )
