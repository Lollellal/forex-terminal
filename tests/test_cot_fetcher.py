"""Tests fuer den CFTC CoT-Fetcher (kein Netzwerk-Zugriff noetig)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cot_fetcher import (
    G7_CURRENCY_FILTERS,
    calculate_net_percentile,
    extract_cot_features,
    filter_g7_currencies,
    save_to_parquet,
)


# ── Test-Fixture ───────────────────────────────────────────────────────────

def make_raw_df(n_weeks: int = 200) -> pd.DataFrame:
    """Erstellt minimalen Mock der CFTC-Rohdaten."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-05", periods=n_weeks, freq="W-TUE")

    markets = {
        "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
        "GBP": "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
        "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
        "CAD": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
        "CHF": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
        "AUD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
        "USD": "U.S. DOLLAR INDEX - ICE FUTURES U.S.",
    }

    rows = []
    for date in dates:
        for currency, market_name in markets.items():
            long_ = int(rng.integers(50_000, 200_000))
            short_ = int(rng.integers(50_000, 200_000))
            rows.append({
                "Market_and_Exchange_Names": market_name,
                "Report_Date_as_YYYY-MM-DD": date.strftime("%Y-%m-%d"),
                "NonComm_Positions_Long_All": long_,
                "NonComm_Positions_Short_All": short_,
                "Change_in_NonComm_Long_All": int(rng.integers(-10_000, 10_000)),
                "Change_in_NonComm_Short_All": int(rng.integers(-10_000, 10_000)),
                "Open_Interest_All": long_ + short_ + int(rng.integers(10_000, 50_000)),
            })

    # Nicht-Waehrungs-Kontrakt — muss herausgefiltert werden
    for date in dates[:5]:
        rows.append({
            "Market_and_Exchange_Names": "WHEAT - CHICAGO BOARD OF TRADE",
            "Report_Date_as_YYYY-MM-DD": date.strftime("%Y-%m-%d"),
            "NonComm_Positions_Long_All": 10_000,
            "NonComm_Positions_Short_All": 5_000,
            "Change_in_NonComm_Long_All": 100,
            "Change_in_NonComm_Short_All": -50,
            "Open_Interest_All": 20_000,
        })

    return pd.DataFrame(rows)


# ── filter_g7_currencies ───────────────────────────────────────────────────

def test_filter_erkennt_alle_g7():
    raw = make_raw_df()
    result = filter_g7_currencies(raw)
    assert set(result["currency"].unique()) == set(G7_CURRENCY_FILTERS.keys())


def test_filter_entfernt_nicht_waehrungen():
    raw = make_raw_df()
    result = filter_g7_currencies(raw)
    assert "WHEAT" not in result["Market_and_Exchange_Names"].str.upper().values


def test_filter_fehlende_spalte_wirft_fehler():
    df = pd.DataFrame({"Falsche_Spalte": ["test"]})
    with pytest.raises(KeyError, match="Market_and_Exchange_Names"):
        filter_g7_currencies(df)


# ── extract_cot_features ───────────────────────────────────────────────────

def test_features_schema():
    raw = make_raw_df()
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)

    expected = {
        "date", "currency", "non_comm_long", "non_comm_short",
        "net_position", "net_change", "long_ratio",
        "open_interest", "change_long", "change_short",
    }
    assert expected.issubset(set(features.columns))


def test_net_position_berechnung():
    raw = make_raw_df()
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    assert (features["net_position"] == features["non_comm_long"] - features["non_comm_short"]).all()


def test_net_change_berechnung():
    raw = make_raw_df()
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    assert (features["net_change"] == features["change_long"] - features["change_short"]).all()


def test_long_ratio_zwischen_0_und_1():
    raw = make_raw_df()
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    assert (features["long_ratio"] >= 0).all()
    assert (features["long_ratio"] <= 1).all()


def test_date_ist_datetime():
    raw = make_raw_df()
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    assert pd.api.types.is_datetime64_any_dtype(features["date"])


# ── calculate_net_percentile ───────────────────────────────────────────────

def test_perzentil_bereich():
    raw = make_raw_df()
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    result = calculate_net_percentile(features)

    assert "net_percentile" in result.columns
    assert (result["net_percentile"] >= 0).all()
    assert (result["net_percentile"] <= 100).all()


def test_erste_zeile_ist_neutral():
    """Erste Zeile pro Waehrung hat keine Historie → Standardwert 50.0."""
    raw = make_raw_df()
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    result = calculate_net_percentile(features)

    first_rows = result.groupby("currency").first()
    assert (first_rows["net_percentile"] == 50.0).all()


def test_extreme_position_hat_hohes_perzentil():
    """Wenn letzter Wert groesser als alle historischen → Perzentil nahe 100."""
    raw = make_raw_df(n_weeks=50)
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)

    # Setze letzten EUR-Wert auf massiven Extremwert
    eur_idx = features[features["currency"] == "EUR"].index
    last_idx = eur_idx[-1]
    features.loc[last_idx, "net_position"] = 10_000_000

    result = calculate_net_percentile(features)
    eur_last = result.loc[last_idx, "net_percentile"]
    assert eur_last == 100.0


# ── save_to_parquet ────────────────────────────────────────────────────────

def test_parquet_wird_erstellt_und_korrekt_geladen():
    raw = make_raw_df()
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    result = calculate_net_percentile(features)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = save_to_parquet(result, Path(tmpdir))
        assert out_path.exists()
        assert out_path.suffix == ".parquet"

        loaded = pd.read_parquet(out_path)
        assert len(loaded) == len(result)
        assert set(loaded.columns) == set(result.columns)


def test_parquet_dateiname_enthaelt_datum():
    raw = make_raw_df(n_weeks=10)
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    result = calculate_net_percentile(features)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = save_to_parquet(result, Path(tmpdir))
        assert "cot_g7_" in out_path.name
