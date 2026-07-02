"""Commands des Empire-Aggregates. Siehe BACKEND_ARCHITECTURE.md §2.2."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from backend.domain.shared.command import Command


@dataclass(frozen=True, kw_only=True)
class CreateEmpireCommand(Command):
    user_id: uuid.UUID
    name: str
