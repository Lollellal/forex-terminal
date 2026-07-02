"""SYNCHRON, in-process aufgerufen vom AllocationLifecycleService vor
ConfirmAllocationCommand und MarkAllocationOpenedCommand. Evaluiert alle
anwendbaren SYNC_GATE-Policies, gibt eine Decision zurück, MUTIERT NICHTS
(reine Lesezugriffe für den Evaluationskontext). Siehe
BACKEND_ARCHITECTURE.md §2.2."""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import Connection, text

from backend.domain.allocation.trade_allocation import TradeAllocation

from .decision import ALLOW, ADJUST, REJECT, RiskGateDecision
from .policies.base import RiskEvaluationContext, RiskPolicy
from .policy_scope_resolver import PolicyScopeResolver

_ACCOUNT_OWNER_SQL = text("SELECT user_id, empire_id FROM core.accounts WHERE id = :id")

_SAME_PAIR_OPEN_SQL = text(
    """
    SELECT EXISTS (
        SELECT 1 FROM core.trade_allocations
        WHERE account_id = :account_id AND pair = :pair AND status IN ('CONFIRMED','OPEN')
          AND id <> :exclude_id
    )
    """
)

_LAST_TWO_CLOSED_SQL = text(
    """
    SELECT realized_r FROM core.trade_allocations
    WHERE account_id = :account_id AND status = 'CLOSED'
    ORDER BY closed_at DESC
    LIMIT 2
    """
)


class RiskGateService:
    def __init__(self, policies: list[RiskPolicy], scope_resolver: PolicyScopeResolver | None = None) -> None:
        self._policies = {policy.policy_key: policy for policy in policies}
        self._scope_resolver = scope_resolver or PolicyScopeResolver()

    def evaluate(
        self, conn: Connection, *, allocation: TradeAllocation, requested_risk_pct: Decimal
    ) -> RiskGateDecision:
        account_row = conn.execute(_ACCOUNT_OWNER_SQL, {"id": str(allocation.account_id)}).one()
        configs = self._scope_resolver.resolve(
            conn,
            user_id=str(account_row.user_id),
            empire_id=str(account_row.empire_id) if account_row.empire_id else None,
            prop_firm_template_id=None,
            account_id=str(allocation.account_id),
            signal_id=str(allocation.signal_id) if allocation.signal_id else None,
            allocation_id=str(allocation.id),
        )
        context = self._build_context(conn, allocation, requested_risk_pct)

        decisions: list[RiskGateDecision] = []
        for config in configs:
            policy = self._policies.get(config.policy_key)
            if policy is None:
                continue
            decision = policy.evaluate(context, config)
            if decision is not None:
                decisions.append(decision)

        return _resolve(decisions, requested_risk_pct)

    def _build_context(
        self, conn: Connection, allocation: TradeAllocation, requested_risk_pct: Decimal
    ) -> RiskEvaluationContext:
        same_pair_open = conn.execute(
            _SAME_PAIR_OPEN_SQL,
            {
                "account_id": str(allocation.account_id),
                "pair": allocation.pair,
                "exclude_id": str(allocation.id),
            },
        ).scalar_one()

        last_two = conn.execute(
            _LAST_TWO_CLOSED_SQL, {"account_id": str(allocation.account_id)}
        ).all()
        last_two_are_losses = len(last_two) == 2 and all(row[0] < 0 for row in last_two)

        return RiskEvaluationContext(
            account_id=allocation.account_id,
            allocation_id=allocation.id,
            pair=allocation.pair,
            requested_risk_pct=requested_risk_pct,
            same_pair_open_exists=bool(same_pair_open),
            last_two_closed_are_losses=last_two_are_losses,
        )


def _resolve(decisions: list[RiskGateDecision], requested_risk_pct: Decimal) -> RiskGateDecision:
    rejects = [d for d in decisions if d.decision_type == REJECT]
    if rejects:
        return rejects[0]

    adjusts = [d for d in decisions if d.decision_type == ADJUST]
    if adjusts:
        return min(adjusts, key=lambda d: d.risk_pct)

    return RiskGateDecision(
        decision_type=ALLOW, risk_pct=requested_risk_pct, reason="keine Policy ausgelöst"
    )
