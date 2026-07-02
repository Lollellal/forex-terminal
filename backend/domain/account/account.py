"""Account-Aggregate. Siehe BACKEND_ARCHITECTURE.md §2.1/§2.2.

Bewusst minimal für Implementierungsschritt 2: nur Balance/Equity/Status.
Risk-Policy-Felder (max_dd_pct etc.) und die Prop-Firm-Stage-Maschine
(CHALLENGE/VERIFICATION/FUNDED) gehören fachlich zum Risk-Gate-Schritt und
werden dort additiv ergänzt, nicht hier vorweggenommen.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from backend.domain.shared.aggregate_root import AggregateRoot
from backend.domain.shared.event_envelope import EventEnvelope

from . import events

VALID_ACCOUNT_TYPES = frozenset({"LIVE", "PROP_FIRM"})
VALID_STATUSES = frozenset({"ACTIVE", "CLOSED"})


class Account(AggregateRoot):
    aggregate_type = "Account"

    def __init__(self, aggregate_id: uuid.UUID, version: int = 0) -> None:
        super().__init__(aggregate_id, version)
        self.user_id: uuid.UUID | None = None
        self.empire_id: uuid.UUID | None = None
        self.account_type: str | None = None
        self.status: str | None = None
        self.balance: Decimal | None = None
        self.equity: Decimal | None = None

    @classmethod
    def create(
        cls,
        aggregate_id: uuid.UUID,
        *,
        user_id: uuid.UUID,
        empire_id: uuid.UUID | None,
        account_type: str,
        initial_balance: Decimal,
        initial_equity: Decimal,
        source: str,
        correlation_id: uuid.UUID,
    ) -> "Account":
        if account_type not in VALID_ACCOUNT_TYPES:
            raise ValueError(f"invalid account_type: {account_type!r}")
        account = cls(aggregate_id)
        account.raise_event(
            events.ACCOUNT_CREATED,
            {
                "user_id": str(user_id),
                "empire_id": str(empire_id) if empire_id else None,
                "account_type": account_type,
                "balance": str(initial_balance),
                "equity": str(initial_equity),
            },
            source=source,
            correlation_id=correlation_id,
        )
        return account

    def update_balance(
        self,
        *,
        balance: Decimal,
        equity: Decimal,
        source: str,
        correlation_id: uuid.UUID,
    ) -> None:
        if self.status == "CLOSED":
            raise ValueError(f"Account {self.id} ist CLOSED — keine Balance-Updates mehr möglich")
        self.raise_event(
            events.ACCOUNT_UPDATED,
            {
                "balance": str(balance),
                "equity": str(equity),
                "status": self.status,
            },
            source=source,
            correlation_id=correlation_id,
        )

    def apply(self, event: EventEnvelope) -> None:
        if event.event_type == events.ACCOUNT_CREATED:
            self.user_id = uuid.UUID(event.payload["user_id"])
            empire_id = event.payload["empire_id"]
            self.empire_id = uuid.UUID(empire_id) if empire_id else None
            self.account_type = event.payload["account_type"]
            self.status = "ACTIVE"
            self.balance = Decimal(event.payload["balance"])
            self.equity = Decimal(event.payload["equity"])
        elif event.event_type == events.ACCOUNT_UPDATED:
            self.balance = Decimal(event.payload["balance"])
            self.equity = Decimal(event.payload["equity"])
            self.status = event.payload["status"]
        else:
            raise ValueError(f"Account kennt event_type {event.event_type!r} nicht")
