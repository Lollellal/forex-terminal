"""Fehlerklassen des Risk Gate."""

from __future__ import annotations

from .decision import RiskGateDecision


class RiskGateRejectedError(Exception):
    """Eine Policy hat REJECT entschieden — der auslösende Command (Confirm/
    Open) darf nicht ausgeführt werden. Trägt die Decision, damit der Aufrufer
    den Grund kennt."""

    def __init__(self, decision: RiskGateDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason)
