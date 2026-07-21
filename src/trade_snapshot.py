"""Baut den unveränderlichen Signal-Snapshot für den Desktop-Cloud-Publish
(v1.1 Mobile Trade Intelligence). Reine Datenextraktion aus bereits
bestehenden lokalen Quellen (signal_journal.json, terminal_data.json,
registrierte WeeklyReports) — keine neue Trading-/Scoring-Logik, kein LLM.

Felder ohne belastbare historische Quelle (overall_score, seasonality_score
als Zahl, primary_drivers) bleiben bewusst None statt erfunden zu werden;
sie sind für eine spätere Erweiterung des Desktop-Publish reserviert.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Eingefrorener Parametersatz, siehe research_execution_costaware.py.
BE_TRADING_DAYS = 5
MAX_HOLD_TRADING_DAYS = 15

_CCY_SECTION_RE_CACHE: dict[str, re.Pattern] = {}


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


def add_trading_days(start: date, n: int) -> date:
    """n Handelstage (Mo-Fr) nach start -- identisch zu
    terminal_server.py::_add_trading_days(), hier zentral fuer alle
    Publish-Konsumenten (Mobile/Webapp) verfuegbar."""
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def extract_currency_section(md_text: str, ccy: str) -> str | None:
    """Zieht den Abschnitt '### {CCY} — ... (Score: ...)' aus einem Weekly-
    Report-Markdown -- identisch zu terminal_server.py::_extract_currency_section()."""
    pattern = _CCY_SECTION_RE_CACHE.get(ccy)
    if pattern is None:
        pattern = re.compile(
            r"###\s*" + re.escape(ccy) + r"\s*—.*?\n(.*?)(?=\n###\s|\n---|\Z)", re.DOTALL
        )
        _CCY_SECTION_RE_CACHE[ccy] = pattern
    match = pattern.search(md_text)
    return match.group(1).strip() if match else None


def build_trade_context(pair: str, entry_date: date, report_md_path: Path | None) -> dict[str, Any]:
    """BE-/Schliess-Datum (reine Kalenderrechnung) + Makro-Begruendung (aus
    dem lokalen Wochenreport-Markdown, falls vorhanden) fuer einen Trade --
    wird Teil des unveraenderlichen signal_snapshot, damit Mobile/Webapp
    dieselbe Information wie das Desktop-Terminal zeigen kann, ohne einen
    eigenen Report-Parsing-Endpunkt zu brauchen."""
    context: dict[str, Any] = {
        "be_date": add_trading_days(entry_date, BE_TRADING_DAYS).isoformat(),
        "close_date": add_trading_days(entry_date, MAX_HOLD_TRADING_DAYS).isoformat(),
        "be_trading_days": BE_TRADING_DAYS,
        "max_hold_trading_days": MAX_HOLD_TRADING_DAYS,
        "why_base_ccy": None, "why_base_section": None,
        "why_quote_ccy": None, "why_quote_section": None,
    }
    if report_md_path is not None and report_md_path.exists() and len(pair) == 6:
        text = report_md_path.read_text(encoding="utf-8")
        base_ccy, quote_ccy = pair[:3], pair[3:]
        context["why_base_ccy"] = base_ccy
        context["why_base_section"] = extract_currency_section(text, base_ccy)
        context["why_quote_ccy"] = quote_ccy
        context["why_quote_section"] = extract_currency_section(text, quote_ccy)
    return context
