"""Reine Koordination: Command entgegennehmen, Aggregate bauen/laden,
speichern. Keine Invarianten hier — die liegen im Aggregate."""

from __future__ import annotations

import uuid

from sqlalchemy import Connection

from .commands import AddJournalNoteCommand, EditJournalNoteCommand
from .journal_note import JournalNote
from .journal_note_repository import JournalNoteRepository


class JournalNoteService:
    def __init__(self, repository: JournalNoteRepository) -> None:
        self._repository = repository

    def add(self, conn: Connection, command: AddJournalNoteCommand) -> JournalNote:
        note = JournalNote.add(
            uuid.uuid4(),
            text=command.text,
            related_allocation_id=command.related_allocation_id,
            related_signal_id=command.related_signal_id,
            attachments=command.attachments,
            source=command.source,
            correlation_id=command.correlation_id,
        )
        self._repository.save(conn, note)
        return note

    def edit(self, conn: Connection, command: EditJournalNoteCommand) -> JournalNote:
        note = self._repository.load(conn, command.note_id)
        note.edit(
            text=command.text,
            attachments=command.attachments,
            source=command.source,
            correlation_id=command.correlation_id,
        )
        self._repository.save(conn, note)
        return note
