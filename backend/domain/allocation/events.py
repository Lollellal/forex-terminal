"""Event-Typen des TradeAllocation-Aggregates. Siehe BACKEND_ARCHITECTURE.md
§2.2. Status-Lifecycle bewusst auf 4 Stufen verkürzt (kein PartialHit/
BreakEven/Cancelled) — die gehören zum Risk-Gate-/Ausführungsschritt, der
hier noch nicht gebaut wird."""

from __future__ import annotations

ALLOCATION_CREATED = "AllocationCreated"
"""Payload: account_id (str), signal_id (str|None), pair (str),
direction ('LONG'|'SHORT'), planned_risk_pct (str), entry_price_planned
(str|None), sl_price (str|None), tp_price (str|None)."""

ALLOCATION_CONFIRMED = "AllocationConfirmed"
"""Payload: applied_risk_pct (str) — vom Risk Gate freigegebenes Risiko,
kann von planned_risk_pct abweichen (ADJUST-Decision)."""

ALLOCATION_OPENED = "AllocationOpened"
"""Payload: opened_at (ISO-String), applied_risk_pct (str) — erneutes
Risk-Gate-Ergebnis zum Open-Zeitpunkt, kann vom Confirm-Wert abweichen.
Kein Preis: OPEN bedeutet "diese Allocation wurde bewusst bestätigt und ist
jetzt aktiv", nicht "der Broker hat zu einem bestimmten Preis gefillt" —
dieses System ist ein Decision-/Research-/Performance-System, kein Order-
Management-System."""

ALLOCATION_CLOSED = "AllocationClosed"
"""Payload: close_reason (str), realized_r (str), closed_at (ISO-String)."""
