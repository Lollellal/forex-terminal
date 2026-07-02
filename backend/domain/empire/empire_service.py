"""Reine Koordination, analog account_service.py."""

from __future__ import annotations

import uuid

from sqlalchemy import Connection

from .commands import CreateEmpireCommand
from .empire import Empire
from .empire_repository import EmpireRepository


class EmpireService:
    def __init__(self, repository: EmpireRepository) -> None:
        self._repository = repository

    def create(self, conn: Connection, command: CreateEmpireCommand) -> Empire:
        empire = Empire.create(
            uuid.uuid4(),
            user_id=command.user_id,
            name=command.name,
            source=command.source,
            correlation_id=command.correlation_id,
        )
        self._repository.save(conn, empire)
        return empire
