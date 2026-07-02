"""TradeAllocation-Aggregate. Siehe BACKEND_ARCHITECTURE.md §2.1/§2.2.

Status-Lifecycle für Implementierungsschritt 3: CREATED -> CONFIRMED -> OPEN
-> CLOSED, strikt linear, keine Sprünge. Risk-Gate-Aufrufe, PartialHit/
BreakEven-Zwischenzustände und die Balance-Reaktion auf AllocationClosed
sind bewusst nicht Teil dieses Aggregates — die kommen mit dem Risk-Gate-
Schritt (siehe DOMAIN_ARCHITECTURE.md: Aggregat kennt keine anderen
Aggregate, reagiert nicht auf sich selbst).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from backend.domain.shared.aggregate_root import AggregateRoot
from backend.domain.shared.event_envelope import EventEnvelope

from . import events

VALID_DIRECTIONS = frozenset({"LONG", "SHORT"})
VALID_CLOSE_REASONS = frozenset({"SL", "TP", "TIME_EXIT", "FORCE_CLOSE", "MANUAL"})

_TRANSITIONS = {
    events.ALLOCATION_CONFIRMED: "CREATED",
    events.ALLOCATION_OPENED: "CONFIRMED",
    events.ALLOCATION_CLOSED: "OPEN",
}


class TradeAllocation(AggregateRoot):
    aggregate_type = "TradeAllocation"

    def __init__(self, aggregate_id: uuid.UUID, version: int = 0) -> None:
        super().__init__(aggregate_id, version)
        self.account_id: uuid.UUID | None = None
        self.signal_id: uuid.UUID | None = None
        self.pair: str | None = None
        self.direction: str | None = None
        self.status: str | None = None
        self.planned_risk_pct: Decimal | None = None
        self.applied_risk_pct: Decimal | None = None
        self.entry_price_planned: Decimal | None = None
        self.sl_price: Decimal | None = None
        self.tp_price: Decimal | None = None
        self.opened_at: datetime | None = None
        self.closed_at: datetime | None = None
        self.close_reason: str | None = None
        self.realized_r: Decimal | None = None
        self.signal_snapshot: dict | None = None

    @classmethod
    def create(
        cls,
        aggregate_id: uuid.UUID,
        *,
        account_id: uuid.UUID,
        pair: str,
        direction: str,
        planned_risk_pct: Decimal,
        signal_id: uuid.UUID | None = None,
        entry_price_planned: Decimal | None = None,
        sl_price: Decimal | None = None,
        tp_price: Decimal | None = None,
        signal_snapshot: dict | None = None,
        source: str,
        correlation_id: uuid.UUID,
    ) -> "TradeAllocation":
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"invalid direction: {direction!r}")
        if planned_risk_pct <= 0:
            raise ValueError("planned_risk_pct muss positiv sein")
        allocation = cls(aggregate_id)
        allocation.raise_event(
            events.ALLOCATION_CREATED,
            {
                "account_id": str(account_id),
                "signal_id": str(signal_id) if signal_id else None,
                "pair": pair,
                "direction": direction,
                "planned_risk_pct": str(planned_risk_pct),
                "entry_price_planned": str(entry_price_planned) if entry_price_planned is not None else None,
                "sl_price": str(sl_price) if sl_price is not None else None,
                "tp_price": str(tp_price) if tp_price is not None else None,
                "signal_snapshot": signal_snapshot,
            },
            source=source,
            correlation_id=correlation_id,
        )
        return allocation

    def confirm(self, *, applied_risk_pct: Decimal, source: str, correlation_id: uuid.UUID) -> None:
        self._require_status(events.ALLOCATION_CONFIRMED)
        if applied_risk_pct <= 0:
            raise ValueError("applied_risk_pct muss positiv sein")
        self.raise_event(
            events.ALLOCATION_CONFIRMED,
            {"applied_risk_pct": str(applied_risk_pct)},
            source=source,
            correlation_id=correlation_id,
        )

    def mark_opened(
        self,
        *,
        applied_risk_pct: Decimal,
        opened_at: datetime | None = None,
        source: str,
        correlation_id: uuid.UUID,
    ) -> None:
        """OPEN bedeutet "bewusst bestätigt und jetzt aktiv" — kein Fill,
        kein Preis. Dieses System ist ein Decision-/Research-/Performance-
        System, kein Order-Management-System."""
        self._require_status(events.ALLOCATION_OPENED)
        if applied_risk_pct <= 0:
            raise ValueError("applied_risk_pct muss positiv sein")
        resolved_opened_at = opened_at or datetime.now(timezone.utc)
        self.raise_event(
            events.ALLOCATION_OPENED,
            {
                "opened_at": resolved_opened_at.isoformat(),
                "applied_risk_pct": str(applied_risk_pct),
            },
            source=source,
            correlation_id=correlation_id,
        )

    def close(
        self,
        *,
        close_reason: str,
        realized_r: Decimal,
        closed_at: datetime | None = None,
        source: str,
        correlation_id: uuid.UUID,
    ) -> None:
        if close_reason not in VALID_CLOSE_REASONS:
            raise ValueError(f"invalid close_reason: {close_reason!r}")
        self._require_status(events.ALLOCATION_CLOSED)
        resolved_closed_at = closed_at or datetime.now(timezone.utc)
        self.raise_event(
            events.ALLOCATION_CLOSED,
            {
                "close_reason": close_reason,
                "realized_r": str(realized_r),
                "closed_at": resolved_closed_at.isoformat(),
            },
            source=source,
            correlation_id=correlation_id,
        )

    def _require_status(self, event_type: str) -> None:
        expected = _TRANSITIONS[event_type]
        if self.status != expected:
            raise ValueError(
                f"TradeAllocation {self.id}: {event_type} erfordert Status {expected!r}, "
                f"aktuell {self.status!r}"
            )

    def apply(self, event: EventEnvelope) -> None:
        if event.event_type == events.ALLOCATION_CREATED:
            self.account_id = uuid.UUID(event.payload["account_id"])
            signal_id = event.payload["signal_id"]
            self.signal_id = uuid.UUID(signal_id) if signal_id else None
            self.pair = event.payload["pair"]
            self.direction = event.payload["direction"]
            self.planned_risk_pct = Decimal(event.payload["planned_risk_pct"])
            self.entry_price_planned = _optional_decimal(event.payload["entry_price_planned"])
            self.sl_price = _optional_decimal(event.payload["sl_price"])
            self.tp_price = _optional_decimal(event.payload["tp_price"])
            self.signal_snapshot = event.payload.get("signal_snapshot")
            self.status = "CREATED"
        elif event.event_type == events.ALLOCATION_CONFIRMED:
            self.applied_risk_pct = Decimal(event.payload["applied_risk_pct"])
            self.status = "CONFIRMED"
        elif event.event_type == events.ALLOCATION_OPENED:
            self.opened_at = datetime.fromisoformat(event.payload["opened_at"])
            self.applied_risk_pct = Decimal(event.payload["applied_risk_pct"])
            self.status = "OPEN"
        elif event.event_type == events.ALLOCATION_CLOSED:
            self.close_reason = event.payload["close_reason"]
            self.realized_r = Decimal(event.payload["realized_r"])
            self.closed_at = datetime.fromisoformat(event.payload["closed_at"])
            self.status = "CLOSED"
        else:
            raise ValueError(f"TradeAllocation kennt event_type {event.event_type!r} nicht")


def _optional_decimal(value: str | None) -> Decimal | None:
    return Decimal(value) if value is not None else None
