"""Commands des JournalNote-Layers. Siehe BACKEND_ARCHITECTURE.md §2.2."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from backend.domain.shared.command import Command


@dataclass(frozen=True, kw_only=True)
class AddJournalNoteCommand(Command):
    text: str
    related_allocation_id: uuid.UUID | None = None
    related_signal_id: uuid.UUID | None = None
    attachments: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class EditJournalNoteCommand(Command):
    note_id: uuid.UUID
    text: str
    attachments: list[str] | None = None
