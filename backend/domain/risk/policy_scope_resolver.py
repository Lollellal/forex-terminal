"""Sammelt für einen Evaluationskontext alle aktiven Policies über die
Scope-Hierarchie GLOBAL→USER→EMPIRE→PROP_FIRM→ACCOUNT→SIGNAL→ALLOCATION und
löst Overrides pro policy_key auf — die spezifischste Ebene gewinnt.
Siehe BACKEND_ARCHITECTURE.md §2.2/§2.4."""

from __future__ import annotations

from sqlalchemy import Connection, text

from .risk_policy import SCOPE_SPECIFICITY, RiskPolicyConfig

_SELECT_APPLICABLE_SQL = text(
    """
    SELECT id, policy_key, name, description, evaluation_mode, scope_type, scope_id,
           priority, enabled, adjusted_risk_pct
    FROM core.risk_policies
    WHERE enabled = TRUE
      AND (
        scope_type = 'GLOBAL'
        OR (scope_type = 'USER' AND scope_id = :user_id)
        OR (scope_type = 'EMPIRE' AND scope_id = :empire_id)
        OR (scope_type = 'PROP_FIRM' AND scope_id = :prop_firm_template_id)
        OR (scope_type = 'ACCOUNT' AND scope_id = :account_id)
        OR (scope_type = 'SIGNAL' AND scope_id = :signal_id)
        OR (scope_type = 'ALLOCATION' AND scope_id = :allocation_id)
      )
    """
)


class PolicyScopeResolver:
    def resolve(
        self,
        conn: Connection,
        *,
        user_id: str | None,
        empire_id: str | None,
        prop_firm_template_id: str | None,
        account_id: str | None,
        signal_id: str | None,
        allocation_id: str | None,
    ) -> list[RiskPolicyConfig]:
        rows = (
            conn.execute(
                _SELECT_APPLICABLE_SQL,
                {
                    "user_id": user_id,
                    "empire_id": empire_id,
                    "prop_firm_template_id": prop_firm_template_id,
                    "account_id": account_id,
                    "signal_id": signal_id,
                    "allocation_id": allocation_id,
                },
            )
            .mappings()
            .all()
        )
        configs = [_row_to_config(row) for row in rows]
        return _resolve_overrides(configs)


def _resolve_overrides(configs: list[RiskPolicyConfig]) -> list[RiskPolicyConfig]:
    """Bei gleichem policy_key auf mehreren Scope-Ebenen gewinnt die
    spezifischere vollständig — kein Kombinieren (BACKEND_ARCHITECTURE.md
    §2.4, Entscheidung 2)."""
    winners: dict[str, RiskPolicyConfig] = {}
    for config in configs:
        current = winners.get(config.policy_key)
        if current is None or SCOPE_SPECIFICITY[config.scope_type] > SCOPE_SPECIFICITY[current.scope_type]:
            winners[config.policy_key] = config
    return sorted(winners.values(), key=lambda c: c.priority)


def _row_to_config(row) -> RiskPolicyConfig:
    return RiskPolicyConfig(
        id=row["id"],
        policy_key=row["policy_key"],
        name=row["name"],
        description=row["description"],
        evaluation_mode=row["evaluation_mode"],
        scope_type=row["scope_type"],
        scope_id=row["scope_id"],
        priority=row["priority"],
        enabled=row["enabled"],
        adjusted_risk_pct=row["adjusted_risk_pct"],
    )
