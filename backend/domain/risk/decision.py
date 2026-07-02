"""RiskGateDecision-Contract. Siehe BACKEND_ARCHITECTURE.md §2.2
(risk_gate_service.py: "gibt Decision zurück (ALLOW/ADJUST/REJECT),
MUTIERT NICHTS")."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

ALLOW = "ALLOW"
ADJUST = "ADJUST"
REJECT = "REJECT"


@dataclass(frozen=True)
class RiskGateDecision:
    decision_type: str
    risk_pct: Decimal | None  # None nur bei REJECT
    reason: str
    triggered_policy: str | None = None
