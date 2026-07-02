"""Event-Typen des Account-Aggregates. Siehe BACKEND_ARCHITECTURE.md §2.2.

Reine Namenskonstanten + Payload-Contracts als TypedDict-Kommentar in den
Docstrings — die tatsächliche Validierung passiert beim Aufbau der
EventEnvelope (siehe account.py), nicht hier.
"""

from __future__ import annotations

ACCOUNT_CREATED = "AccountCreated"
"""Payload: empire_id (str|None), account_type ('LIVE'|'PROP_FIRM'),
balance (str), equity (str)."""

ACCOUNT_UPDATED = "AccountUpdated"
"""Payload: balance (str), equity (str), status ('ACTIVE'|'CLOSED')."""
