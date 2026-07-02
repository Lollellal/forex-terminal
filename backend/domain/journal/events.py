"""Event-Typen des JournalNote-Aggregates. Siehe BACKEND_ARCHITECTURE.md
§2.2."""

from __future__ import annotations

JOURNAL_NOTE_ADDED = "JournalNoteAdded"
"""Payload: related_allocation_id (str|None), related_signal_id (str|None),
text (str), attachments (list[str])."""

JOURNAL_NOTE_EDITED = "JournalNoteEdited"
"""Payload: text (str), attachments (list[str])."""
