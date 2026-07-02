"""Policy-Interface: reine Entscheidungsfunktion, zustandslos, testbar ohne
DB (BACKEND_ARCHITECTURE.md §2.2: "Policy ... evaluate(context) -> Decision,
zustandslos"). Der Gate baut den Context per DB-Abfrage, die Policy selbst
sieht nur fertige Werte."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from backend.domain.risk.decision import RiskGateDecision
from backend.domain.risk.risk_policy import RiskPolicyConfig


@dataclass(frozen=True)
class RiskEvaluationContext:
    account_id: uuid.UUID
    allocation_id: uuid.UUID
    pair: str
    requested_risk_pct: Decimal
    same_pair_open_exists: bool
    last_two_closed_are_losses: bool


class RiskPolicy(Protocol):
    policy_key: str

    def evaluate(
        self, context: RiskEvaluationContext, config: RiskPolicyConfig
    ) -> RiskGateDecision | None: ...
