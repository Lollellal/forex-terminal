"""Tests fuer den Kalender-Backfill-Runner (kein Netzwerk-Zugriff noetig)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.calendar_backfill import (
    _chunk_months,
    _load_checkpoint,
    _month_bounds,
    _month_range,
    _save_checkpoint,
    run_backfill,
    run_backfill_batched,
)
from src.data.calendar_fetcher import BotBlockedError


# ── _month_range ──────────────────────────────────────────────────────────

def test_month_range_innerhalb_eines_jahres():
    assert _month_range("2015-01", "2015-03") == ["2015-01", "2015-02", "2015-03"]


def test_month_range_ueber_jahreswechsel():
    assert _month_range("2015-11", "2016-02") == ["2015-11", "2015-12", "2016-01", "2016-02"]


def test_month_range_einzelner_monat():
    assert _month_range("2015-06", "2015-06") == ["2015-06"]


# ── _month_bounds ─────────────────────────────────────────────────────────

def test_month_bounds_normaler_monat():
    assert _month_bounds("2015-06") == ("2015-06-01", "2015-06-30")


def test_month_bounds_dezember():
    assert _month_bounds("2015-12") == ("2015-12-01", "2015-12-31")


def test_month_bounds_februar_schaltjahr():
    assert _month_bounds("2016-02") == ("2016-02-01", "2016-02-29")


def test_month_bounds_februar_kein_schaltjahr():
    assert _month_bounds("2015-02") == ("2015-02-01", "2015-02-28")


# ── Checkpoint ────────────────────────────────────────────────────────────

def test_checkpoint_laden_wenn_datei_fehlt():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = _load_checkpoint(Path(tmpdir) / "nicht_vorhanden.json")
        assert checkpoint == {"done": [], "failed": []}


def test_checkpoint_speichern_und_laden_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.json"
        _save_checkpoint(path, {"done": ["2015-01"], "failed": []})
        loaded = _load_checkpoint(path)
        assert loaded == {"done": ["2015-01"], "failed": []}


# ── run_backfill (gemockt) ──────────────────────────────────────────────

def _mock_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": "2015-06-01", "time": "8:30", "currency": "USD",
         "event_name": "Nonfarm Payrolls", "impact": "high",
         "actual": "200K", "forecast": "190K", "previous": "180K"}
        for _ in range(n)
    ])


@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_alle_monate_erfolgreich(mock_fetch, mock_sleep):
    mock_fetch.return_value = _mock_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        checkpoint_path = Path(tmpdir) / "checkpoint.json"

        summary = run_backfill(
            "2015-01", "2015-03", out_dir=out_dir, checkpoint_path=checkpoint_path,
            rate_limit_seconds=0,
        )

        assert summary["done"] == ["2015-01", "2015-02", "2015-03"]
        assert summary["failed"] == []
        assert len(list(out_dir.glob("*.parquet"))) == 3

        checkpoint = _load_checkpoint(checkpoint_path)
        assert checkpoint["done"] == ["2015-01", "2015-02", "2015-03"]


@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_ueberspringt_bereits_erledigte_monate(mock_fetch, mock_sleep):
    mock_fetch.return_value = _mock_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        checkpoint_path = Path(tmpdir) / "checkpoint.json"
        _save_checkpoint(checkpoint_path, {"done": ["2015-01"], "failed": []})

        summary = run_backfill(
            "2015-01", "2015-03", out_dir=out_dir, checkpoint_path=checkpoint_path,
            rate_limit_seconds=0,
        )

        assert summary["skipped"] == ["2015-01"]
        assert summary["done"] == ["2015-02", "2015-03"]
        # fetch_month wurde nur fuer die zwei neuen Monate aufgerufen
        assert mock_fetch.call_count == 2


@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_markiert_leere_monate_als_failed(mock_fetch, mock_sleep):
    mock_fetch.return_value = pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        checkpoint_path = Path(tmpdir) / "checkpoint.json"

        summary = run_backfill(
            "2015-01", "2015-01", out_dir=out_dir, checkpoint_path=checkpoint_path,
            rate_limit_seconds=0,
        )

        assert summary["failed"] == ["2015-01"]
        assert summary["done"] == []
        checkpoint = _load_checkpoint(checkpoint_path)
        assert "2015-01" in checkpoint["failed"]
        assert len(list(out_dir.glob("*.parquet"))) == 0


@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_resume_nach_teilweisem_fehlschlag(mock_fetch, mock_sleep):
    """Simuliert Abbruch nach Monat 1 (fehlgeschlagen), dann erfolgreichen Resume-Lauf."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        checkpoint_path = Path(tmpdir) / "checkpoint.json"

        # Erster Lauf: Monat 1 schlaegt fehl, Monat 2 klappt
        mock_fetch.side_effect = [pd.DataFrame(), _mock_df()]
        run_backfill(
            "2015-01", "2015-02", out_dir=out_dir, checkpoint_path=checkpoint_path,
            rate_limit_seconds=0,
        )
        checkpoint = _load_checkpoint(checkpoint_path)
        assert checkpoint["done"] == ["2015-02"]
        assert checkpoint["failed"] == ["2015-01"]

        # Zweiter Lauf (Resume): Monat 1 klappt jetzt, Monat 2 wird uebersprungen
        mock_fetch.side_effect = [_mock_df()]
        summary = run_backfill(
            "2015-01", "2015-02", out_dir=out_dir, checkpoint_path=checkpoint_path,
            rate_limit_seconds=0,
        )
        assert summary["done"] == ["2015-01"]
        assert summary["skipped"] == ["2015-02"]

        checkpoint = _load_checkpoint(checkpoint_path)
        assert set(checkpoint["done"]) == {"2015-01", "2015-02"}
        assert checkpoint["failed"] == []


@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_exception_wird_abgefangen(mock_fetch, mock_sleep):
    """Ein Monat, der eine Exception wirft, darf den gesamten Lauf nicht abbrechen."""
    mock_fetch.side_effect = [RuntimeError("boom"), _mock_df()]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        checkpoint_path = Path(tmpdir) / "checkpoint.json"

        summary = run_backfill(
            "2015-01", "2015-02", out_dir=out_dir, checkpoint_path=checkpoint_path,
            rate_limit_seconds=0,
        )

        assert summary["failed"] == ["2015-01"]
        assert summary["done"] == ["2015-02"]


@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_rate_limit_wird_zwischen_monaten_aufgerufen(mock_fetch, mock_sleep):
    mock_fetch.return_value = _mock_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        run_backfill(
            "2015-01", "2015-03",
            out_dir=Path(tmpdir) / "out",
            checkpoint_path=Path(tmpdir) / "checkpoint.json",
            rate_limit_seconds=5.0,
        )
        # 3 Monate → 2 Pausen zwischen den Requests, keine nach dem letzten
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(5.0)


# ── Bot-Block-Handling ──────────────────────────────────────────────────

@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_stoppt_batch_sofort_bei_bot_block(mock_fetch, mock_sleep):
    """Regression: Bot-Block (429) darf NICHT wie 'keine Daten' behandelt werden —
    der blockierte Monat bleibt unmarkiert, weitere Monate werden nicht mehr versucht."""
    mock_fetch.side_effect = [_mock_df(), BotBlockedError("429")]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        checkpoint_path = Path(tmpdir) / "checkpoint.json"

        summary = run_backfill(
            "2015-01", "2015-03", out_dir=out_dir, checkpoint_path=checkpoint_path,
            rate_limit_seconds=0,
        )

        assert summary["done"] == ["2015-01"]
        assert summary["blocked_at"] == "2015-02"
        # 2015-02 (blockiert) und 2015-03 (nicht mehr versucht) sind NICHT 'failed'
        assert summary["failed"] == []
        assert mock_fetch.call_count == 2   # 2015-03 wurde gar nicht erst versucht

        checkpoint = _load_checkpoint(checkpoint_path)
        assert checkpoint["done"] == ["2015-01"]
        assert "2015-02" not in checkpoint["failed"]
        assert "2015-03" not in checkpoint["failed"]


@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_resume_nach_bot_block(mock_fetch, mock_sleep):
    """Nach einem Block muss ein erneuter Lauf den blockierten Monat erneut versuchen
    (nicht ueberspringen, da er nie als 'done' oder 'failed' markiert wurde)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        checkpoint_path = Path(tmpdir) / "checkpoint.json"

        mock_fetch.side_effect = [_mock_df(), BotBlockedError("429")]
        run_backfill(
            "2015-01", "2015-02", out_dir=out_dir, checkpoint_path=checkpoint_path,
            rate_limit_seconds=0,
        )

        mock_fetch.side_effect = [_mock_df()]
        summary = run_backfill(
            "2015-01", "2015-02", out_dir=out_dir, checkpoint_path=checkpoint_path,
            rate_limit_seconds=0,
        )
        assert summary["done"] == ["2015-02"]
        assert summary["skipped"] == ["2015-01"]
        assert summary["blocked_at"] is None


# ── _chunk_months ─────────────────────────────────────────────────────────

def test_chunk_months_teilt_in_gleichmaessige_batches():
    months = [f"2015-{m:02d}" for m in range(1, 13)]
    chunks = _chunk_months(months, 5)
    assert chunks == [
        months[0:5], months[5:10], months[10:12],
    ]


def test_chunk_months_ein_batch_wenn_kleiner_als_batch_size():
    months = ["2015-01", "2015-02"]
    assert _chunk_months(months, 20) == [months]


# ── run_backfill_batched ────────────────────────────────────────────────

@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_batched_mehrere_batches_mit_pause(mock_fetch, mock_sleep):
    mock_fetch.return_value = _mock_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = run_backfill_batched(
            "2015-01", "2015-06",
            out_dir=Path(tmpdir) / "out",
            checkpoint_path=Path(tmpdir) / "checkpoint.json",
            rate_limit_seconds=0,
            batch_size=2,
            batch_pause_seconds=99,
        )

        assert summary["batches_run"] == 3
        assert len(summary["done"]) == 6
        assert summary["blocked_at"] is None
        # Pause zwischen Batches: 2 Pausen fuer 3 Batches (99s), plus 0s-Pausen
        # innerhalb der Batches (rate_limit_seconds=0) — pruefe nur die Batch-Pause
        assert mock_sleep.call_args_list.count(((99,),)) == 2


@patch("src.data.calendar_backfill.time.sleep")
@patch("src.data.calendar_backfill.fetch_month")
def test_run_backfill_batched_stoppt_gesamten_lauf_bei_block(mock_fetch, mock_sleep):
    """Ein Block in Batch 1 darf Batch 2 nicht mehr anlaufen lassen."""
    mock_fetch.side_effect = [_mock_df(), BotBlockedError("429")]

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = run_backfill_batched(
            "2015-01", "2015-06",
            out_dir=Path(tmpdir) / "out",
            checkpoint_path=Path(tmpdir) / "checkpoint.json",
            rate_limit_seconds=0,
            batch_size=2,
            batch_pause_seconds=99,
        )

        assert summary["batches_run"] == 1
        assert summary["blocked_at"] == "2015-02"
        assert summary["done"] == ["2015-01"]
        # fetch_month wurde nur 2x aufgerufen (Monat 1 + der blockierte Monat 2) —
        # Batch 2/3 (Maerz-Juni) wurden gar nicht erst gestartet
        assert mock_fetch.call_count == 2
