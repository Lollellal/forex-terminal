"""Two Consecutive Losses — reduziert das Risiko, wenn die letzten zwei
geschlossenen Allocations dieses Accounts beide Verlust waren."""

from __future__ import annotations

from backend.domain.risk.decision import ADJUST, RiskGateDecision
from backend.domain.risk.risk_policy import RiskPolicyConfig

from .base import RiskEvaluationContext


class ConsecutiveLossesPolicy:
    policy_key = "consecutive-losses"

    def evaluate(
        self, context: RiskEvaluationContext, config: RiskPolicyConfig
    ) -> RiskGateDecision | None:
        if not context.last_two_closed_are_losses:
            return None
        return RiskGateDecision(
            decision_type=ADJUST,
            risk_pct=config.adjusted_risk_pct,
            reason=f"Account {context.account_id}: letzte 2 Trades waren Verlust",
            triggered_policy=self.policy_key,
        )
