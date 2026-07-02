"""Schritt 10: Desktop-to-Cloud Publish Layer.

Manuelles Skript (kein Scheduler, siehe project_legacy_data_locations),
liest die bestehende Desktop-Datenhaltung (``data/empire.json`` +
``data/signal_journal.json``) und spielt sie als Commands gegen das neue
Event-Sourcing-Backend (Account + TradeAllocation Lifecycle) durch.

Idempotent über eine Mapping-Datei (Legacy-ID -> Backend-UUID,
``backend/scripts/legacy_import_map.json``): wiederholte Läufe legen
nichts doppelt an, sondern laden den bereits gemappten Aggregate-Zustand
und führen ihn nur so weit fort wie nötig (z.B. neu WIN/LOSS gewordene
Signale schließen offene Allocations nach).

Aufruf: ``python -m backend.scripts.publish_legacy`` im Projekt-Root.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from backend.domain.account.account_service import AccountService
from backend.domain.account.account_repository import AccountRepository
from backend.domain.account.commands import CreateAccountCommand
from backend.domain.allocation.allocation_lifecycle_service import AllocationLifecycleService
from backend.domain.allocation.allocation_repository import AllocationRepository
from backend.domain.allocation.commands import (
    CloseAllocationCommand,
    ConfirmAllocationCommand,
    CreateAllocationCommand,
    MarkAllocationOpenedCommand,
)
from backend.domain.risk.policies.consecutive_losses_policy import ConsecutiveLossesPolicy
from backend.domain.risk.policies.same_pair_policy import SamePairOpenPolicy
from backend.domain.risk.risk_gate_service import RiskGateService
from backend.infrastructure.db import get_engine
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMPIRE_PATH = PROJECT_ROOT / "data" / "empire.json"
SIGNAL_JOURNAL_PATH = PROJECT_ROOT / "data" / "signal_journal.json"
MAPPING_PATH = Path(__file__).resolve().parent / "legacy_import_map.json"

# Fixer, dauerhafter User (core.users, label='David') — siehe
# project_backend_dev_environment. Kein Auth-System vorhanden, daher hart
# hinterlegt statt nachgeschlagen.
REAL_USER_ID = uuid.UUID("28182fce-d78e-4585-b6be-a33914f7271c")

_DIRECTION_MAP = {"long": "LONG", "short": "SHORT"}


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_mapping() -> dict:
    if MAPPING_PATH.exists():
        return _load_json(MAPPING_PATH)
    return {"accounts": {}, "allocations": {}}


def _save_mapping(mapping: dict) -> None:
    MAPPING_PATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _risk_gate() -> RiskGateService:
    return RiskGateService([SamePairOpenPolicy(), ConsecutiveLossesPolicy()])


def publish() -> None:
    empire = _load_json(EMPIRE_PATH)
    signals_by_id = {s["id"]: s for s in _load_json(SIGNAL_JOURNAL_PATH)}
    mapping = _load_mapping()

    engine = get_engine()
    event_store = EventStore()
    account_service = AccountService(AccountRepository(event_store))
    account_repo = AccountRepository(event_store)
    allocation_repo = AllocationRepository(event_store)
    lifecycle_service = AllocationLifecycleService(allocation_repo, _risk_gate())

    for legacy_account in empire["accounts"]:
        legacy_account_id = legacy_account["id"]
        with engine.begin() as conn:
            if legacy_account_id in mapping["accounts"]:
                account_id = uuid.UUID(mapping["accounts"][legacy_account_id])
                account_repo.load(conn, account_id)  # nur Existenzcheck
                print(f"Account {legacy_account_id} bereits importiert -> {account_id}")
            else:
                initial_balance = Decimal(str(legacy_account["initial_balance"]))
                account = account_service.create(
                    conn,
                    CreateAccountCommand(
                        user_id=REAL_USER_ID,
                        account_type="PROP_FIRM",
                        initial_balance=initial_balance,
                        initial_equity=initial_balance,
                        source="system",
                    ),
                )
                account_id = account.id
                mapping["accounts"][legacy_account_id] = str(account_id)
                _save_mapping(mapping)
                print(f"Account {legacy_account_id} neu angelegt -> {account_id}")

        _publish_allocations(
            conn_factory=engine,
            lifecycle_service=lifecycle_service,
            allocation_repo=allocation_repo,
            mapping=mapping,
            legacy_allocations=[
                a for a in empire["allocations"] if a["account_id"] == legacy_account_id
            ],
            backend_account_id=account_id,
            signals_by_id=signals_by_id,
        )

    with engine.begin() as conn:
        ProjectionRunner(event_store).catch_up(conn)
    print("Projektionen aktualisiert.")


def _publish_allocations(
    *,
    conn_factory,
    lifecycle_service: AllocationLifecycleService,
    allocation_repo: AllocationRepository,
    mapping: dict,
    legacy_allocations: list[dict],
    backend_account_id: uuid.UUID,
    signals_by_id: dict,
) -> None:
    ordered = sorted(legacy_allocations, key=lambda a: _parse_dt(a["created_at"]))

    for legacy_alloc in ordered:
        legacy_id = legacy_alloc["id"]
        signal = signals_by_id.get(legacy_alloc["signal_id"])
        if signal is None:
            print(f"Allocation {legacy_id}: Signal {legacy_alloc['signal_id']} nicht gefunden, übersprungen")
            continue

        direction = _DIRECTION_MAP[signal["direction"].lower()]
        pair = signal["pair"]
        planned_risk_pct = Decimal(str(legacy_alloc["risk_pct"]))
        created_at = _parse_dt(legacy_alloc["created_at"])

        with conn_factory.begin() as conn:
            if legacy_id in mapping["allocations"]:
                allocation_id = uuid.UUID(mapping["allocations"][legacy_id])
                current = allocation_repo.load(conn, allocation_id)
            else:
                created = lifecycle_service.create(
                    conn,
                    CreateAllocationCommand(
                        account_id=backend_account_id,
                        pair=pair,
                        direction=direction,
                        planned_risk_pct=planned_risk_pct,
                        source="system",
                    ),
                )
                allocation_id = created.id
                mapping["allocations"][legacy_id] = str(allocation_id)
                _save_mapping(mapping)
                current = created

            if current.status == "CREATED":
                current = lifecycle_service.confirm(
                    conn, ConfirmAllocationCommand(allocation_id=allocation_id, source="system")
                )
            if current.status == "CONFIRMED":
                current = lifecycle_service.mark_opened(
                    conn,
                    MarkAllocationOpenedCommand(
                        allocation_id=allocation_id, opened_at=created_at, source="system"
                    ),
                )
            if current.status == "OPEN" and signal["status"] in ("WIN", "LOSS"):
                if signal["status"] == "WIN":
                    realized_r = Decimal(str(signal["return_pct"]))
                else:
                    # Legacy-System zieht bei LOSS immer den vollen Risikobetrag ab,
                    # unabhängig vom gespeicherten return_pct (siehe
                    # compute_account_stats in src/empire.py) — realized_r = -1R.
                    realized_r = Decimal("-1")
                # Legacy-Daten kennen den konkreten Exit-Mechanismus (SL/TP/Zeit)
                # nicht — MANUAL ist ehrlicher als ein geratener Wert.
                close_reason = "MANUAL"
                settled_at = signal.get("settled_at")
                closed_at = _parse_dt(settled_at) if settled_at else created_at
                current = lifecycle_service.close(
                    conn,
                    CloseAllocationCommand(
                        allocation_id=allocation_id,
                        close_reason=close_reason,
                        realized_r=realized_r,
                        closed_at=closed_at,
                        source="system",
                    ),
                )

        print(f"Allocation {legacy_id} ({pair} {direction}) -> {allocation_id}: Status {current.status}")


if __name__ == "__main__":
    publish()
