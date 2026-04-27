"""Tests fuer den FRED Makrodaten-Fetcher (kein Netzwerk-Zugriff noetig)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.macro_fetcher import (
    ALL_INDICATORS,
    G7_CURRENCIES,
    _get_api_key,
    fetch_indicator,
    fetch_series,
    load_latest,
    save_to_parquet,
)


# ── Hilfsfunktionen ────────────────────────────────────────────────────────

def make_fred_response(series_id: str, n: int = 10, missing_last: bool = False) -> dict:
    """Erstellt ein minimales Mock-FRED-JSON-Response-Objekt."""
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    observations = []
    for i, date in enumerate(dates):
        value = "." if (missing_last and i == n - 1) else str(1.5 + i * 0.1)
        observations.append({"date": date.strftime("%Y-%m-%d"), "value": value})
    return {"observations": observations}


def make_mock_response(series_id: str, n: int = 10, missing_last: bool = False) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = make_fred_response(series_id, n, missing_last)
    return mock


# ── _get_api_key ───────────────────────────────────────────────────────────

def test_api_key_fehlt_wirft_fehler():
    with patch.dict("os.environ", {}, clear=True):
        with patch("src.data.macro_fetcher.os.getenv", return_value=""):
            with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
                _get_api_key()


def test_api_key_placeholder_wirft_fehler():
    with patch("src.data.macro_fetcher.os.getenv", return_value="your_fred_api_key_here"):
        with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
            _get_api_key()


def test_api_key_valide_wird_zurueckgegeben():
    with patch("src.data.macro_fetcher.os.getenv", return_value="abc123valid"):
        assert _get_api_key() == "abc123valid"


# ── fetch_series ───────────────────────────────────────────────────────────

def test_fetch_series_parst_antwort_korrekt():
    with patch("requests.get", return_value=make_mock_response("FEDFUNDS", n=5)):
        result = fetch_series("FEDFUNDS", "testkey")

    assert isinstance(result, pd.Series)
    assert len(result) == 5
    assert pd.api.types.is_float_dtype(result)


def test_fetch_series_punkt_wird_zu_none():
    """FRED kodiert fehlende Werte als '.' — diese muessen None werden."""
    with patch("requests.get", return_value=make_mock_response("FEDFUNDS", n=5, missing_last=True)):
        result = fetch_series("FEDFUNDS", "testkey")

    assert pd.isna(result.iloc[-1])
    assert not pd.isna(result.iloc[0])


def test_fetch_series_index_ist_datetime():
    with patch("requests.get", return_value=make_mock_response("UNRATE", n=6)):
        result = fetch_series("UNRATE", "testkey")

    assert pd.api.types.is_datetime64_any_dtype(result.index)


def test_fetch_series_falsches_format_wirft_fehler():
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"error": "Not found"}

    with patch("requests.get", return_value=mock):
        with pytest.raises(ValueError, match="Unerwartetes FRED-Antwortformat"):
            fetch_series("INVALID", "testkey")


def test_fetch_series_http_fehler_wird_weitergegeben():
    mock = MagicMock()
    mock.raise_for_status.side_effect = Exception("HTTP 400")

    with patch("requests.get", return_value=mock):
        with pytest.raises(Exception, match="HTTP 400"):
            fetch_series("FEDFUNDS", "testkey")


# ── fetch_indicator ────────────────────────────────────────────────────────

def test_fetch_indicator_schema():
    """DataFrame muss die erwarteten Spalten enthalten."""
    with patch("requests.get", return_value=make_mock_response("X", n=12)):
        df = fetch_indicator("interest_rate", {"USD": "FEDFUNDS", "EUR": "ECBDFR"}, "testkey")

    expected_cols = {"date", "currency", "indicator", "series_id", "value"}
    assert expected_cols.issubset(set(df.columns))


def test_fetch_indicator_alle_waehrungen_enthalten():
    """Alle 7 G7-Waehrungen muessen im Ergebnis auftauchen."""
    series_map = {c: f"SERIES_{c}" for c in G7_CURRENCIES}

    with patch("requests.get", return_value=make_mock_response("X", n=5)):
        df = fetch_indicator("cpi_yoy", series_map, "testkey")

    assert set(df["currency"].unique()) == G7_CURRENCIES


def test_fetch_indicator_indikator_name_korrekt():
    with patch("requests.get", return_value=make_mock_response("X", n=5)):
        df = fetch_indicator("unemployment", {"USD": "UNRATE"}, "testkey")

    assert (df["indicator"] == "unemployment").all()


def test_fetch_indicator_fehlertoleranz():
    """Schlaegt eine Waehrung fehl, werden die anderen trotzdem geladen."""
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Netzwerkfehler simuliert")
        return make_mock_response("X", n=5)

    with patch("requests.get", side_effect=side_effect):
        df = fetch_indicator(
            "interest_rate",
            {"USD": "FEDFUNDS", "EUR": "ECBDFR", "GBP": "BOERUKM"},
            "testkey",
        )

    # USD fehlgeschlagen → 2 Waehrungen uebrig
    assert df["currency"].nunique() == 2
    assert "USD" not in df["currency"].values


def test_fetch_indicator_alle_fehler_wirft_runtime_error():
    mock = MagicMock()
    mock.raise_for_status.side_effect = Exception("Alle kaputt")

    with patch("requests.get", return_value=mock):
        with pytest.raises(RuntimeError, match="Keine Daten"):
            fetch_indicator("gdp_qoq", {"USD": "A191RL1Q225SBEA"}, "testkey")


def test_fetch_indicator_fehlende_werte_werden_entfernt():
    """Zeilen mit value=None (FRED '.') duerfen nicht im Ergebnis stehen."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "observations": [
            {"date": "2020-01-01", "value": "1.5"},
            {"date": "2020-02-01", "value": "."},
            {"date": "2020-03-01", "value": "2.0"},
        ]
    }

    with patch("requests.get", return_value=mock):
        df = fetch_indicator("interest_rate", {"USD": "FEDFUNDS"}, "testkey")

    assert df["value"].notna().all()
    assert len(df) == 2


# ── save_to_parquet & load_latest ─────────────────────────────────────────

def make_indicator_df(indicator: str = "interest_rate", n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    return pd.DataFrame({
        "date": list(dates) * 2,
        "value": [1.5] * (n * 2),
        "currency": ["USD"] * n + ["EUR"] * n,
        "indicator": [indicator] * (n * 2),
        "series_id": ["FEDFUNDS"] * n + ["ECBDFR"] * n,
    })


def test_save_parquet_erstellt_datei():
    df = make_indicator_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_to_parquet(df, "interest_rate", Path(tmpdir))
        assert path.exists()
        assert path.suffix == ".parquet"


def test_save_parquet_dateiname_konvention():
    df = make_indicator_df("cpi_yoy")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_to_parquet(df, "cpi_yoy", Path(tmpdir))
        assert path.name.startswith("macro_cpi_yoy_")


def test_save_parquet_rundreise():
    """Daten muessen nach Write/Read identisch sein."""
    df = make_indicator_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_to_parquet(df, "interest_rate", Path(tmpdir))
        loaded = pd.read_parquet(path)

    assert len(loaded) == len(df)
    assert set(loaded.columns) == set(df.columns)
    assert loaded["value"].sum() == pytest.approx(df["value"].sum())


def test_save_parquet_erstellt_verzeichnis():
    df = make_indicator_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "neu" / "tief"
        path = save_to_parquet(df, "unemployment", nested)
        assert path.exists()


def test_load_latest_laedt_neueste_datei():
    df = make_indicator_df("gdp_qoq")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        save_to_parquet(df, "gdp_qoq", out_dir)
        loaded = load_latest("gdp_qoq", out_dir)

    assert len(loaded) == len(df)


def test_load_latest_fehler_wenn_keine_datei():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="interest_rate"):
            load_latest("interest_rate", Path(tmpdir))
