"""Same Pair Already Open — reduziert das Risiko, wenn auf demselben Account
bereits eine bestätigte/offene Allocation auf demselben Pair existiert."""

from __future__ import annotations

from backend.domain.risk.decision import ADJUST, RiskGateDecision
from backend.domain.risk.risk_policy import RiskPolicyConfig

from .base import RiskEvaluationContext


class SamePairOpenPolicy:
    policy_key = "same-pair-open"

    def evaluate(
        self, context: RiskEvaluationContext, config: RiskPolicyConfig
    ) -> RiskGateDecision | None:
        if not context.same_pair_open_exists:
            return None
        return RiskGateDecision(
            decision_type=ADJUST,
            risk_pct=config.adjusted_risk_pct,
            reason=f"{context.pair} bereits offen auf Account {context.account_id}",
            triggered_policy=self.policy_key,
        )
