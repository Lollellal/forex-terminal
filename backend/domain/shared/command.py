"""Command-Basisklasse. Commands sind Absichten (Intent), keine Fakten —
siehe DOMAIN_ARCHITECTURE.md §0.6 (Command/Event-Trennung).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .event_envelope import VALID_SOURCES


@dataclass(frozen=True, kw_only=True)
class Command:
    """Basis für alle Commands.

    Konkrete Commands (z.B. ConfirmAllocationCommand) sind eigene
    ``@dataclass(frozen=True, kw_only=True)``-Klassen, die von dieser Klasse
    erben und ihre fachlichen Felder ergänzen. ``kw_only=True`` ist Pflicht,
    weil sonst das dataclass-Vererbungsproblem zuschlägt: Pflichtfelder der
    Subklasse dürften sonst nicht nach den Default-Feldern hier stehen.
    """

    command_id: uuid.UUID = field(default_factory=uuid.uuid4)
    correlation_id: uuid.UUID = field(default_factory=uuid.uuid4)
    causation_id: uuid.UUID | None = None
    issued_by_user_id: uuid.UUID | None = None
    issued_by_device_id: uuid.UUID | None = None
    source: str = "system"
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(f"invalid command source: {self.source!r}")
