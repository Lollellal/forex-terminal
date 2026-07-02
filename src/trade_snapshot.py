"""Baut den unveränderlichen Signal-Snapshot für den Desktop-Cloud-Publish
(v1.1 Mobile Trade Intelligence). Reine Datenextraktion aus bereits
bestehenden lokalen Quellen (signal_journal.json, terminal_data.json,
registrierte WeeklyReports) — keine neue Trading-/Scoring-Logik, kein LLM.

Felder ohne belastbare historische Quelle (overall_score, seasonality_score
als Zahl, primary_drivers) bleiben bewusst None statt erfunden zu werden;
sie sind für eine spätere Erweiterung des Desktop-Publish reserviert.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def build_signal_snapshot(signal: dict[str, Any], terminal_data: dict[str, Any]) -> dict[str, Any]:
    created_at = _parse_dt(signal["created_at"])
    return {
        "ml_score": signal.get("ml_score"),
        "quality": signal.get("quality"),
        "edge": signal.get("edge_5y"),
        "alignment": signal.get("alignment"),
        "combo_key": signal.get("combo_key"),
        "regime": _regime_at(terminal_data, created_at),
        "overall_score": None,
        "seasonality_score": None,
        "primary_drivers": None,
        "weekly_report_id": None,
        "report_week": None,
    }


def _regime_at(terminal_data: dict[str, Any], at: datetime) -> str | None:
    """Nächstliegender terminal_data['history']-Eintrag zum gegebenen
    Zeitpunkt (letzter Eintrag mit timestamp <= at); fällt auf das aktuell
    hinterlegte Regime zurück, falls die Historie leer ist oder ausschließlich
    aus späteren Einträgen besteht."""
    history = terminal_data.get("history") or []
    preceding = [h for h in history if _parse_dt(h["timestamp"]) <= at]
    if preceding:
        nearest = max(preceding, key=lambda h: _parse_dt(h["timestamp"]))
        return (nearest.get("regime") or {}).get("regime")
    current = terminal_data.get("regime")
    return current.get("regime") if current else None


def resolve_weekly_report(
    signal_date: date, reports: list[dict[str, Any]]
) -> tuple[str | None, str | None]:
    """Mechanischer Datumsabgleich: der Report, dessen period_start/period_end
    den Signal-Zeitpunkt überdeckt. Kein inhaltlicher Bezug, reiner
    Zeitraum-Match gegen bereits registrierte Reports."""
    for report in reports:
        period_start = _as_date(report["period_start"])
        period_end = _as_date(report["period_end"])
        if period_start <= signal_date <= period_end:
            iso = period_start.isocalendar()
            return str(report["id"]), f"KW{iso.week}/{iso.year}"
    return None, None


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def with_weekly_report(snapshot: dict[str, Any], report_id: str | None, report_week: str | None) -> dict[str, Any]:
    return {**snapshot, "weekly_report_id": report_id, "report_week": report_week}
