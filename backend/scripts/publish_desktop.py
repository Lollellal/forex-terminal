"""Desktop-Cloud-Publish (v1.1 Mobile Trade Intelligence, vormals Schritt 10
"publish_legacy.py"). Der einzige Sync-Pfad Desktop -> Cloud: Desktop bleibt
alleinige Research-/Berechnungsquelle (ML/Scanner/Trade-Auswahl/Signal-
Snapshot/Weekly Report), dieses Skript veröffentlicht das Ergebnis
unverändert ins Backend. Mobile konsumiert ausschließlich über die API.

Pipeline pro Lauf:
    Accounts publizieren
    -> Weekly Reports registrieren+veröffentlichen (inkl. Executive-Summary-
       Extraktion aus der .md-Quelle, mechanisch, kein LLM)
    -> Allocations publizieren (pro Allocation: Signal-Snapshot bauen,
       gegen bereits registrierte Reports auflösen, unveränderlich einfrieren)
    -> aktuelles Marktregime als Cache-Wert veröffentlichen
    -> Projektionen aktualisieren

Idempotent über eine Mapping-Datei (Legacy-ID -> Backend-UUID,
``backend/scripts/desktop_publish_map.json``): wiederholte Läufe legen
nichts doppelt an.

signal_snapshot/summary werden nur beim jeweiligen CREATE-Event gesetzt und
sind danach unveränderlich (Event Sourcing) — ein erneuter Lauf kann sie
nachträglich NICHT mehr ergänzen, nur neu erzeugte Allocations/Reports
bekommen sie.

Aufruf: ``python -m backend.scripts.publish_desktop`` im Projekt-Root.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text

from backend.domain.account.account_repository import AccountRepository
from backend.domain.account.account_service import AccountService
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
from backend.domain.weekly_report.commands import PublishWeeklyReportCommand, RegisterWeeklyReportCommand
from backend.domain.weekly_report.weekly_report_repository import WeeklyReportRepository
from backend.domain.weekly_report.weekly_report_service import WeeklyReportService
from backend.infrastructure.db import get_engine
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner
from backend.infrastructure.storage import WeeklyReportStorage
from src.trade_snapshot import build_signal_snapshot, resolve_weekly_report, with_weekly_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMPIRE_PATH = PROJECT_ROOT / "data" / "empire.json"
SIGNAL_JOURNAL_PATH = PROJECT_ROOT / "data" / "signal_journal.json"
TERMINAL_DATA_PATH = PROJECT_ROOT / "terminal_data.json"
REPORTS_DIR = PROJECT_ROOT / "Research weekly reports"
REPORTS_MD_DIR = PROJECT_ROOT / "reports"
MAPPING_PATH = Path(__file__).resolve().parent / "desktop_publish_map.json"

# Fixer, dauerhafter User (core.users, label='David') — siehe
# project_backend_dev_environment. Kein Auth-System vorhanden, daher hart
# hinterlegt statt nachgeschlagen.
REAL_USER_ID = uuid.UUID("28182fce-d78e-4585-b6be-a33914f7271c")

_DIRECTION_MAP = {"long": "LONG", "short": "SHORT"}
_REPORT_FILENAME_RE = re.compile(r"^weekly_(\d{4})_W(\d{2})$")
_EXEC_SUMMARY_RE = re.compile(
    r"##\s*1\.\s*Executive Summary\s*\n+(.*?)(?:\n---|\n##\s|\Z)", re.DOTALL
)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_mapping() -> dict:
    if MAPPING_PATH.exists():
        return _load_json(MAPPING_PATH)
    return {"accounts": {}, "allocations": {}, "reports": {}}


def _save_mapping(mapping: dict) -> None:
    MAPPING_PATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _risk_gate() -> RiskGateService:
    return RiskGateService([SamePairOpenPolicy(), ConsecutiveLossesPolicy()])


def _extract_executive_summary(md_path: Path) -> str | None:
    """Mechanisches Section-Parsing: der Text zwischen der Überschrift
    '## 1. Executive Summary' und der nächsten Überschrift/Trennlinie. Kein
    LLM, keine Zusammenfassung — der Text steht bereits so in der Quelle."""
    if not md_path.exists():
        return None
    match = _EXEC_SUMMARY_RE.search(md_path.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else None


def publish() -> None:
    empire = _load_json(EMPIRE_PATH)
    signals_by_id = {s["id"]: s for s in _load_json(SIGNAL_JOURNAL_PATH)}
    terminal_data = _load_json(TERMINAL_DATA_PATH) if TERMINAL_DATA_PATH.exists() else {}
    mapping = _load_mapping()

    engine = get_engine()
    event_store = EventStore()
    account_service = AccountService(AccountRepository(event_store))
    account_repo = AccountRepository(event_store)
    allocation_repo = AllocationRepository(event_store)
    lifecycle_service = AllocationLifecycleService(allocation_repo, _risk_gate())
    report_service = WeeklyReportService(WeeklyReportRepository(event_store), WeeklyReportStorage())

    _publish_weekly_reports(engine, report_service, mapping)
    known_reports = _fetch_known_reports(engine)

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
            terminal_data=terminal_data,
            known_reports=known_reports,
        )

    with engine.begin() as conn:
        _publish_market_snapshot(conn, terminal_data)
        ProjectionRunner(event_store).catch_up(conn)
    print("Projektionen aktualisiert.")


def _publish_weekly_reports(engine, report_service: WeeklyReportService, mapping: dict) -> None:
    if not REPORTS_DIR.exists():
        return
    for pdf_path in sorted(REPORTS_DIR.glob("weekly_*.pdf")):
        stem = pdf_path.stem
        match = _REPORT_FILENAME_RE.match(stem)
        if match is None:
            continue
        if stem in mapping["reports"]:
            print(f"Report {stem} bereits registriert -> {mapping['reports'][stem]}")
            continue

        year, week = int(match.group(1)), int(match.group(2))
        period_start = date.fromisocalendar(year, week, 1)
        period_end = date.fromisocalendar(year, week, 7)
        summary = _extract_executive_summary(REPORTS_MD_DIR / f"{stem}.md")

        with engine.begin() as conn:
            report = report_service.register(
                conn,
                RegisterWeeklyReportCommand(
                    user_id=REAL_USER_ID,
                    period_start=period_start,
                    period_end=period_end,
                    content=pdf_path.read_bytes(),
                    summary=summary,
                    source="system",
                ),
            )
            report = report_service.publish(
                conn, PublishWeeklyReportCommand(report_id=report.id, source="system")
            )
        mapping["reports"][stem] = str(report.id)
        _save_mapping(mapping)
        print(f"Report {stem} registriert+veröffentlicht -> {report.id}")


def _fetch_known_reports(engine) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, period_start, period_end FROM projections.weekly_reports")
        ).mappings().all()
    return [dict(row) for row in rows]


def _publish_market_snapshot(conn, terminal_data: dict) -> None:
    regime = terminal_data.get("regime")
    if not regime or not regime.get("regime"):
        return
    conn.execute(
        text(
            """
            INSERT INTO projections.market_snapshot (id, regime, vix, yield_curve, updated_at)
            VALUES ('current', :regime, :vix, :yield_curve, now())
            ON CONFLICT (id) DO UPDATE SET
                regime = EXCLUDED.regime, vix = EXCLUDED.vix,
                yield_curve = EXCLUDED.yield_curve, updated_at = now()
            """
        ),
        {"regime": regime.get("regime"), "vix": regime.get("vix"), "yield_curve": regime.get("yield_curve")},
    )
    print(f"Market-Snapshot aktualisiert -> {regime.get('regime')}")


def _publish_allocations(
    *,
    conn_factory,
    lifecycle_service: AllocationLifecycleService,
    allocation_repo: AllocationRepository,
    mapping: dict,
    legacy_allocations: list[dict],
    backend_account_id: uuid.UUID,
    signals_by_id: dict,
    terminal_data: dict,
    known_reports: list[dict],
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
                snapshot = build_signal_snapshot(signal, terminal_data)
                report_id, report_week = resolve_weekly_report(
                    date.fromisoformat(signal["date"]), known_reports
                )
                snapshot = with_weekly_report(snapshot, report_id, report_week)
                created = lifecycle_service.create(
                    conn,
                    CreateAllocationCommand(
                        account_id=backend_account_id,
                        pair=pair,
                        direction=direction,
                        planned_risk_pct=planned_risk_pct,
                        signal_snapshot=snapshot,
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
