"""Orchestriert Create -> Confirm -> Open -> Close. Siehe
BACKEND_ARCHITECTURE.md §2.2: reine Koordination, kein eigener Zustand,
keine Invarianten (die liegen im Aggregate). Ruft den Risk Gate synchron vor
Confirm und vor Open auf (kein direkter Policy-Aufruf aus dem Aggregate
selbst — Trennung von Invariante und Entscheidung). Bei REJECT wird der
State-Übergang gar nicht erst versucht."""

from __future__ import annotations

import uuid

from sqlalchemy import Connection

from backend.domain.risk.decision import REJECT
from backend.domain.risk.exceptions import RiskGateRejectedError
from backend.domain.risk.risk_gate_service import RiskGateService

from .allocation_repository import AllocationRepository
from .commands import (
    CloseAllocationCommand,
    ConfirmAllocationCommand,
    CreateAllocationCommand,
    MarkAllocationOpenedCommand,
)
from .trade_allocation import TradeAllocation


class AllocationLifecycleService:
    def __init__(self, repository: AllocationRepository, risk_gate: RiskGateService) -> None:
        self._repository = repository
        self._risk_gate = risk_gate

    def create(self, conn: Connection, command: CreateAllocationCommand) -> TradeAllocation:
        allocation = TradeAllocation.create(
            uuid.uuid4(),
            account_id=command.account_id,
            pair=command.pair,
            direction=command.direction,
            planned_risk_pct=command.planned_risk_pct,
            signal_id=command.signal_id,
            entry_price_planned=command.entry_price_planned,
            sl_price=command.sl_price,
            tp_price=command.tp_price,
            signal_snapshot=command.signal_snapshot,
            source=command.source,
            correlation_id=command.correlation_id,
        )
        self._repository.save(conn, allocation)
        return allocation

    def confirm(self, conn: Connection, command: ConfirmAllocationCommand) -> TradeAllocation:
        allocation = self._repository.load(conn, command.allocation_id)
        decision = self._risk_gate.evaluate(
            conn, allocation=allocation, requested_risk_pct=allocation.planned_risk_pct
        )
        if decision.decision_type == REJECT:
            raise RiskGateRejectedError(decision)
        allocation.confirm(
            applied_risk_pct=decision.risk_pct, source=command.source, correlation_id=command.correlation_id
        )
        self._repository.save(conn, allocation)
        return allocation

    def mark_opened(
        self, conn: Connection, command: MarkAllocationOpenedCommand
    ) -> TradeAllocation:
        allocation = self._repository.load(conn, command.allocation_id)
        decision = self._risk_gate.evaluate(
            conn, allocation=allocation, requested_risk_pct=allocation.applied_risk_pct
        )
        if decision.decision_type == REJECT:
            raise RiskGateRejectedError(decision)
        allocation.mark_opened(
            applied_risk_pct=decision.risk_pct,
            opened_at=command.opened_at,
            source=command.source,
            correlation_id=command.correlation_id,
        )
        self._repository.save(conn, allocation)
        return allocation

    def close(self, conn: Connection, command: CloseAllocationCommand) -> TradeAllocation:
        allocation = self._repository.load(conn, command.allocation_id)
        allocation.close(
            close_reason=command.close_reason,
            realized_r=command.realized_r,
            closed_at=command.closed_at,
            source=command.source,
            correlation_id=command.correlation_id,
        )
        self._repository.save(conn, allocation)
        return allocation
