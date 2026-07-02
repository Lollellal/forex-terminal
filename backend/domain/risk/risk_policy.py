"""RiskPolicy-Konfiguration (Registry-Eintrag aus core.risk_policies).

Bewusst kein Event-Sourced Aggregate: In diesem Schritt gibt es keine
Commands, die Policies zur Laufzeit anlegen/ändern (das wäre eigener
Scope) — die zwei Policies werden per Migration geseedet. RiskPolicyConfig
ist daher ein reines, unveränderliches Lesemodell einer Zeile aus
core.risk_policies. Siehe BACKEND_ARCHITECTURE.md §2.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

VALID_SCOPE_TYPES = (
    "GLOBAL",
    "USER",
    "EMPIRE",
    "PROP_FIRM",
    "ACCOUNT",
    "SIGNAL",
    "ALLOCATION",
)
# Aufsteigende Spezifität — bestimmt, welche Config bei gleichem policy_key
# auf mehreren Scope-Ebenen gewinnt (BACKEND_ARCHITECTURE.md §2.4,
# Entscheidung 2: "gewinnt die spezifischere").
SCOPE_SPECIFICITY = {scope_type: rank for rank, scope_type in enumerate(VALID_SCOPE_TYPES)}


@dataclass(frozen=True)
class RiskPolicyConfig:
    id: str
    policy_key: str
    name: str
    description: str
    evaluation_mode: str
    scope_type: str
    scope_id: str | None
    priority: int
    enabled: bool
    adjusted_risk_pct: Decimal | None
