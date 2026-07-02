"""Tests fuer src.trade_snapshot — Signal-Snapshot-Builder (v1.1 Mobile Trade Intelligence)."""

from datetime import date

from src.trade_snapshot import build_signal_snapshot, resolve_weekly_report, with_weekly_report

SIGNAL = {
    "id": "abc123",
    "created_at": "2026-06-27T14:54:53.560512",
    "date": "2026-06-29",
    "pair": "AUDCAD",
    "direction": "Short",
    "quality": "VALID",
    "edge_5y": "Weak Edge",
    "alignment": "CONTRARY",
    "combo_key": "VALID|Weak Edge|Short|CONTRARY",
    "ml_score": -0.364,
}

TERMINAL_DATA = {
    "regime": {"regime": "TREND", "vix": 18.4, "yield_curve": 0.31},
    "history": [
        {"timestamp": "2026-06-29T15:52:36.919002+00:00", "regime": {"regime": "TREND"}},
        {"timestamp": "2026-06-22T10:00:00+00:00", "regime": {"regime": "RISK_ON"}},
        {"timestamp": "2026-06-15T10:00:00+00:00", "regime": {"regime": "RISK_OFF"}},
    ],
}


def test_build_signal_snapshot_extracts_structured_fields():
    snapshot = build_signal_snapshot(SIGNAL, TERMINAL_DATA)

    assert snapshot["ml_score"] == -0.364
    assert snapshot["quality"] == "VALID"
    assert snapshot["edge"] == "Weak Edge"
    assert snapshot["alignment"] == "CONTRARY"
    assert snapshot["combo_key"] == "VALID|Weak Edge|Short|CONTRARY"


def test_build_signal_snapshot_leaves_unavailable_fields_none():
    snapshot = build_signal_snapshot(SIGNAL, TERMINAL_DATA)

    assert snapshot["overall_score"] is None
    assert snapshot["seasonality_score"] is None
    assert snapshot["primary_drivers"] is None
    assert snapshot["weekly_report_id"] is None
    assert snapshot["report_week"] is None


def test_regime_uses_nearest_preceding_history_entry():
    # Signal created_at (2026-06-27) liegt zwischen den history-Eintraegen
    # vom 22. (RISK_ON) und 29. (TREND) -> naechstliegender VORHERIGER ist der 22.
    snapshot = build_signal_snapshot(SIGNAL, TERMINAL_DATA)
    assert snapshot["regime"] == "RISK_ON"


def test_regime_falls_back_to_current_when_no_preceding_history():
    signal = {**SIGNAL, "created_at": "2026-06-01T00:00:00"}
    snapshot = build_signal_snapshot(signal, TERMINAL_DATA)
    assert snapshot["regime"] == "TREND"


def test_regime_falls_back_to_current_when_history_empty():
    snapshot = build_signal_snapshot(SIGNAL, {"regime": {"regime": "RANGE"}, "history": []})
    assert snapshot["regime"] == "RANGE"


def test_regime_none_when_nothing_available():
    snapshot = build_signal_snapshot(SIGNAL, {})
    assert snapshot["regime"] is None


def test_resolve_weekly_report_matches_covering_period():
    reports = [
        {"id": "report-1", "period_start": date(2026, 6, 22), "period_end": date(2026, 6, 28)},
        {"id": "report-2", "period_start": date(2026, 6, 15), "period_end": date(2026, 6, 21)},
    ]
    report_id, report_week = resolve_weekly_report(date(2026, 6, 25), reports)
    assert report_id == "report-1"
    assert report_week == "KW26/2026"


def test_resolve_weekly_report_accepts_string_dates():
    reports = [{"id": "report-1", "period_start": "2026-06-22", "period_end": "2026-06-28"}]
    report_id, _ = resolve_weekly_report(date(2026, 6, 25), reports)
    assert report_id == "report-1"


def test_resolve_weekly_report_no_match_returns_none_tuple():
    reports = [{"id": "report-1", "period_start": date(2026, 6, 22), "period_end": date(2026, 6, 28)}]
    report_id, report_week = resolve_weekly_report(date(2026, 7, 10), reports)
    assert report_id is None
    assert report_week is None


def test_with_weekly_report_merges_without_mutating_input():
    base = build_signal_snapshot(SIGNAL, TERMINAL_DATA)
    merged = with_weekly_report(base, "report-1", "KW26/2026")

    assert merged["weekly_report_id"] == "report-1"
    assert merged["report_week"] == "KW26/2026"
    assert base["weekly_report_id"] is None  # Original unveraendert
