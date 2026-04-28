"""Tests fuer das Feature Engineering Modul (kein Netzwerk-/Disk-Zugriff noetig)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import (
    G7_CURRENCIES,
    CB_INFLATION_TARGET,
    COT_EXTREME_HIGH,
    COT_EXTREME_LOW,
    _parse_numeric,
    _wide_to_long,
    _macro_to_daily,
    _cot_to_daily,
    build_date_spine,
    compute_group_a,
    compute_group_b,
    compute_group_e,
    compute_group_f,
    merge_all_features,
    save_to_parquet,
)


# ── Test-Fixtures ──────────────────────────────────────────────────────────

START = "2020-01-01"
END   = "2020-06-30"   # kurze Periode damit Tests schnell sind


def make_date_spine(start: str = START, end: str = END) -> pd.DataFrame:
    return build_date_spine(start=start, end=end)


def make_macro_df(
    indicator: str,
    start: str = "2018-01-01",  # laenger als Spine fuer Trend-Berechnungen
    end: str = END,
    base_value: float = 2.0,
    currencies: list[str] | None = None,
) -> pd.DataFrame:
    """Erstellt synthetische Makrodaten im Format des macro_fetchers."""
    if currencies is None:
        currencies = G7_CURRENCIES
    dates = pd.date_range(start, end, freq="MS")  # Monatlich, 1. des Monats
    rows = []
    for i, date in enumerate(dates):
        for j, currency in enumerate(currencies):
            rows.append({
                "date": date,
                "value": base_value + i * 0.05 + j * 0.1,
                "currency": currency,
                "indicator": indicator,
                "series_id": f"FAKE_{currency}",
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_cot_df(
    n_weeks: int = 60,
    start: str = "2019-01-01",
    currencies: list[str] | None = None,
) -> pd.DataFrame:
    """Erstellt synthetische CoT-Daten im Format des cot_fetchers."""
    if currencies is None:
        currencies = G7_CURRENCIES
    rng = np.random.default_rng(99)
    dates = pd.date_range(start, periods=n_weeks, freq="W-TUE")
    rows = []
    for date in dates:
        for currency in currencies:
            long_  = int(rng.integers(50_000, 200_000))
            short_ = int(rng.integers(50_000, 200_000))
            net    = long_ - short_
            rows.append({
                "date": date,
                "currency": currency,
                "non_comm_long":  long_,
                "non_comm_short": short_,
                "net_position":   net,
                "net_change":     int(rng.integers(-10_000, 10_000)),
                "net_percentile": float(rng.uniform(5, 95)),
                "long_ratio":     long_ / (long_ + short_),
                "change_long":    int(rng.integers(-5_000, 5_000)),
                "change_short":   int(rng.integers(-5_000, 5_000)),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_calendar_df(currencies: list[str] | None = None) -> pd.DataFrame:
    """Erstellt synthetischen Wirtschaftskalender."""
    if currencies is None:
        currencies = ["USD", "EUR", "GBP"]
    rows = []
    for currency in currencies:
        rows.append({
            "date": pd.Timestamp("2020-02-15"),
            "currency": currency,
            "event_name": "CPI y/y",
            "impact": "high",
            "forecast": "2.5%",
            "previous": "2.3%",
        })
        rows.append({
            "date": pd.Timestamp("2020-03-18"),
            "currency": currency,
            "event_name": "Interest Rate Decision",
            "impact": "high",
            "forecast": "1.0%",
            "previous": "1.25%",
        })
    return pd.DataFrame(rows)


# ── _parse_numeric ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("175K",   175_000.0),
    ("1.5M",  1_500_000.0),
    ("2.5B",  2_500_000_000.0),
    ("4.25%",        4.25),
    ("-0.5%",        -0.5),
    ("1,234.5",    1234.5),
    ("0",             0.0),
])
def test_parse_numeric_bekannte_formate(raw, expected):
    assert _parse_numeric(raw) == pytest.approx(expected)


def test_parse_numeric_leer_gibt_nan():
    assert np.isnan(_parse_numeric(""))
    assert np.isnan(_parse_numeric(None))
    assert np.isnan(_parse_numeric("n/a"))


# ── build_date_spine ──────────────────────────────────────────────────────

def test_date_spine_schema():
    spine = build_date_spine(start=START, end=END)
    assert "date" in spine.columns
    assert "currency" in spine.columns


def test_date_spine_alle_waehrungen():
    spine = build_date_spine(start=START, end=END)
    assert set(spine["currency"].unique()) == set(G7_CURRENCIES)


def test_date_spine_nur_handelstage():
    spine = build_date_spine(start="2020-01-01", end="2020-01-07")
    # 2020-01-01 (Mi), 02 (Do), 03 (Fr) — 04/05 Wochenende
    dates = pd.DatetimeIndex(spine["date"].unique())
    assert all(d.weekday() < 5 for d in dates)   # 0=Mo, 4=Fr


def test_date_spine_zeilen_pro_waehrung():
    spine = build_date_spine(start="2020-01-02", end="2020-01-03")
    # 2 Handelstage × 7 Waehrungen = 14 Zeilen
    assert len(spine) == 2 * len(G7_CURRENCIES)


# ── _macro_to_daily ────────────────────────────────────────────────────────

def test_macro_to_daily_liefert_alle_waehrungen():
    macro = make_macro_df("interest_rate")
    spine = make_date_spine()
    wide = _macro_to_daily(macro, "interest_rate", spine)
    for cur in G7_CURRENCIES:
        assert cur in wide.columns


def test_macro_to_daily_forward_fill():
    """Monatsdaten muessen auf alle Handelstage des Monats forward-gefilled sein."""
    macro = make_macro_df("interest_rate")
    spine = make_date_spine()
    wide = _macro_to_daily(macro, "interest_rate", spine)
    usd = wide["USD"].dropna()
    # An einem mittleren Handelstag im Februar muss ein Wert vorhanden sein
    feb_15 = pd.Timestamp("2020-02-15")
    if feb_15 in wide.index:
        assert not np.isnan(wide.loc[feb_15, "USD"])


def test_macro_to_daily_unbekannter_indikator_gibt_nan():
    macro = make_macro_df("interest_rate")
    spine = make_date_spine()
    wide = _macro_to_daily(macro, "NICHT_EXISTENT", spine)
    assert wide.isna().all().all()


# ── _wide_to_long ─────────────────────────────────────────────────────────

def test_wide_to_long_schema():
    macro = make_macro_df("interest_rate")
    spine = make_date_spine()
    wide = _macro_to_daily(macro, "interest_rate", spine)
    long_df = _wide_to_long(wide, "a_interest_rate")
    assert set(long_df.columns) == {"date", "currency", "a_interest_rate"}


def test_wide_to_long_anzahl_zeilen():
    macro = make_macro_df("interest_rate")
    spine = make_date_spine()
    wide = _macro_to_daily(macro, "interest_rate", spine)
    long_df = _wide_to_long(wide, "test_col")
    # wide hat len(spine.date.unique()) × len(G7) Zeilen
    assert len(long_df) == len(wide) * len(G7_CURRENCIES)


# ── compute_group_a ────────────────────────────────────────────────────────

def test_group_a_schema():
    macro = make_macro_df("interest_rate")
    spine = make_date_spine()
    result = compute_group_a(macro, pd.DataFrame(), spine)
    expected = {
        "a_interest_rate", "a_rate_hist_avg_3y", "a_rate_dev_from_avg",
        "a_rate_spread_vs_usd", "a_rate_expected_change",
    }
    assert expected.issubset(set(result.columns))


def test_group_a_usd_spread_ist_null():
    """Zinsdifferenz USD vs USD muss immer 0 sein."""
    macro = make_macro_df("interest_rate")
    spine = make_date_spine()
    result = compute_group_a(macro, pd.DataFrame(), spine)
    usd_spread = result.loc[result["currency"] == "USD", "a_rate_spread_vs_usd"].dropna()
    assert (usd_spread.abs() < 1e-9).all()


def test_group_a_spread_korrekte_richtung():
    """Wenn EUR-Zins > USD-Zins, muss EUR-Spread positiv sein."""
    # Erstelle unterschiedliche Zinssaetze: EUR = 4.0, USD = 2.0
    rows = []
    for d in pd.date_range("2018-01-01", "2020-06-30", freq="MS"):
        rows.append({"date": d, "value": 2.0, "currency": "USD",
                     "indicator": "interest_rate", "series_id": "FEDFUNDS"})
        rows.append({"date": d, "value": 4.0, "currency": "EUR",
                     "indicator": "interest_rate", "series_id": "ECBDFR"})
    macro = pd.DataFrame(rows)
    macro["date"] = pd.to_datetime(macro["date"])

    spine = make_date_spine()
    result = compute_group_a(macro, pd.DataFrame(), spine)
    eur_spread = result.loc[result["currency"] == "EUR", "a_rate_spread_vs_usd"].dropna()
    assert (eur_spread > 0).all(), "EUR hat hoehere Zinsen als USD → Spread muss positiv sein"


def test_group_a_rate_dev_from_avg():
    """rate_dev_from_avg = interest_rate - hist_avg_3y."""
    macro = make_macro_df("interest_rate")
    spine = make_date_spine()
    result = compute_group_a(macro, pd.DataFrame(), spine)
    diff = (result["a_interest_rate"] - result["a_rate_hist_avg_3y"] - result["a_rate_dev_from_avg"]).dropna()
    assert (diff.abs() < 1e-9).all()


def test_group_a_rate_expected_change_aus_kalender():
    """a_rate_expected_change muss aus Kalender-Event berechnet werden."""
    macro = make_macro_df("interest_rate", base_value=1.25)
    spine = make_date_spine()
    # Kalender-Event: forecast = 1.0%, forecast < current (1.25%) → expected change < 0
    cal = pd.DataFrame([{
        "date": pd.Timestamp("2020-03-18"),
        "currency": "USD",
        "event_name": "Interest Rate Decision",
        "impact": "high",
        "forecast": "1.0%",
        "previous": "1.25%",
    }])
    result = compute_group_a(macro, cal, spine)
    # Am Tag des Events oder kurz davor sollte a_rate_expected_change < 0
    ev_rows = result[
        (result["currency"] == "USD")
        & (result["date"] == pd.Timestamp("2020-03-18"))
        & result["a_rate_expected_change"].notna()
    ]
    if not ev_rows.empty:
        assert ev_rows["a_rate_expected_change"].iloc[0] < 0


# ── compute_group_b ────────────────────────────────────────────────────────

def test_group_b_schema():
    macro = make_macro_df("cpi_yoy")
    spine = make_date_spine()
    result = compute_group_b(macro, pd.DataFrame(), spine)
    expected = {
        "b_cpi_yoy", "b_cpi_vs_target", "b_cpi_trend_3m",
        "b_cpi_trend_6m", "b_cpi_vs_usd", "b_cpi_surprise",
    }
    assert expected.issubset(set(result.columns))


def test_group_b_cpi_vs_target():
    """b_cpi_vs_target = b_cpi_yoy - 2.0 (CB-Ziel)."""
    macro = make_macro_df("cpi_yoy", base_value=5.0)
    spine = make_date_spine()
    result = compute_group_b(macro, pd.DataFrame(), spine)
    diff = (result["b_cpi_vs_target"] - (result["b_cpi_yoy"] - CB_INFLATION_TARGET)).dropna()
    assert (diff.abs() < 1e-9).all()


def test_group_b_usd_vs_usd_ist_null():
    """b_cpi_vs_usd fuer USD muss 0 sein."""
    macro = make_macro_df("cpi_yoy")
    spine = make_date_spine()
    result = compute_group_b(macro, pd.DataFrame(), spine)
    usd_vs_usd = result.loc[result["currency"] == "USD", "b_cpi_vs_usd"].dropna()
    assert (usd_vs_usd.abs() < 1e-9).all()


def test_group_b_cpi_vs_usd_richtung():
    """Wenn EUR-CPI > USD-CPI, muss b_cpi_vs_usd fuer EUR positiv sein."""
    rows = []
    for d in pd.date_range("2018-01-01", "2020-06-30", freq="MS"):
        rows.append({"date": d, "value": 2.0, "currency": "USD",
                     "indicator": "cpi_yoy", "series_id": "CPI_USD"})
        rows.append({"date": d, "value": 4.0, "currency": "EUR",
                     "indicator": "cpi_yoy", "series_id": "CPI_EUR"})
    macro = pd.DataFrame(rows)
    macro["date"] = pd.to_datetime(macro["date"])

    spine = make_date_spine()
    result = compute_group_b(macro, pd.DataFrame(), spine)
    eur_vs_usd = result.loc[result["currency"] == "EUR", "b_cpi_vs_usd"].dropna()
    assert (eur_vs_usd > 0).all()


def test_group_b_cpi_trend_3m_vorzeichen():
    """Wenn CPI steigt, muss cpi_trend_3m positiv sein (nach genuegend Verlauf)."""
    rows = []
    # CPI steigt stetig von 2018 bis 2020
    for i, d in enumerate(pd.date_range("2018-01-01", "2020-06-30", freq="MS")):
        rows.append({"date": d, "value": 1.0 + i * 0.1, "currency": "USD",
                     "indicator": "cpi_yoy", "series_id": "CPI_USD"})
    macro = pd.DataFrame(rows)
    macro["date"] = pd.to_datetime(macro["date"])

    spine = make_date_spine()
    result = compute_group_b(macro, pd.DataFrame(), spine)
    # Im Spine-Zeitraum (2020) sollte Trend > 0, da CPI steigt
    usd_trend = result.loc[
        (result["currency"] == "USD") & result["b_cpi_trend_3m"].notna(),
        "b_cpi_trend_3m"
    ]
    assert (usd_trend > 0).all()


def test_group_b_cpi_surprise_aus_kalender():
    """b_cpi_surprise muss nach CPI-Kalender-Event gesetzt sein."""
    macro = make_macro_df("cpi_yoy")
    spine = make_date_spine()
    cal = pd.DataFrame([{
        "date": pd.Timestamp("2020-02-15"),
        "currency": "USD",
        "event_name": "CPI y/y",
        "impact": "high",
        "forecast": "2.5%",
        "previous": "2.8%",  # previous > forecast → positiver Surprise-Proxy
    }])
    result = compute_group_b(macro, cal, spine)
    # Nach dem Event sollte b_cpi_surprise gesetzt sein
    after_event = result.loc[
        (result["currency"] == "USD") & (result["date"] > pd.Timestamp("2020-02-15")),
        "b_cpi_surprise"
    ].dropna()
    assert len(after_event) > 0


# ── compute_group_e ────────────────────────────────────────────────────────

def test_group_e_schema():
    cot = make_cot_df()
    spine = make_date_spine()
    result = compute_group_e(cot, spine)
    expected = {
        "e_cot_net_position", "e_cot_trend_4w", "e_cot_long_ratio",
        "e_cot_net_percentile", "e_cot_weekly_change",
        "e_cot_flow_acceleration", "e_cot_extreme_flag",
    }
    assert expected.issubset(set(result.columns))


def test_group_e_extreme_flag_high():
    """Perzentil > 90 → extreme_flag = +1."""
    cot = make_cot_df()
    spine = make_date_spine()
    # Setze USD-Perzentil explizit auf 95
    cot.loc[cot["currency"] == "USD", "net_percentile"] = 95.0
    result = compute_group_e(cot, spine)
    usd_flags = result.loc[
        (result["currency"] == "USD") & result["e_cot_extreme_flag"].notna(),
        "e_cot_extreme_flag"
    ]
    assert (usd_flags == 1.0).all()


def test_group_e_extreme_flag_low():
    """Perzentil < 10 → extreme_flag = -1."""
    cot = make_cot_df()
    spine = make_date_spine()
    cot.loc[cot["currency"] == "EUR", "net_percentile"] = 5.0
    result = compute_group_e(cot, spine)
    eur_flags = result.loc[
        (result["currency"] == "EUR") & result["e_cot_extreme_flag"].notna(),
        "e_cot_extreme_flag"
    ]
    assert (eur_flags == -1.0).all()


def test_group_e_extreme_flag_neutral():
    """Perzentil zwischen 10 und 90 → extreme_flag = 0."""
    cot = make_cot_df()
    spine = make_date_spine()
    cot.loc[cot["currency"] == "GBP", "net_percentile"] = 50.0
    result = compute_group_e(cot, spine)
    gbp_flags = result.loc[
        (result["currency"] == "GBP") & result["e_cot_extreme_flag"].notna(),
        "e_cot_extreme_flag"
    ]
    assert (gbp_flags == 0.0).all()


def test_group_e_net_position_werte():
    """e_cot_net_position muss auf Nicht-NaN-Tagen einen Wert haben."""
    cot = make_cot_df()
    spine = make_date_spine()
    result = compute_group_e(cot, spine)
    usd_pos = result.loc[result["currency"] == "USD", "e_cot_net_position"].dropna()
    assert len(usd_pos) > 0


def test_group_e_leerer_cot_gibt_nur_nan():
    """Leerer CoT-Input → alle e_*-Spalten NaN."""
    empty_cot = pd.DataFrame(columns=[
        "date", "currency", "net_position", "net_percentile",
        "net_change", "long_ratio",
    ])
    spine = make_date_spine()
    result = compute_group_e(empty_cot, spine)
    e_cols = [c for c in result.columns if c.startswith("e_")]
    for col in e_cols:
        assert result[col].isna().all(), f"{col} sollte komplett NaN sein"


def test_group_e_long_ratio_grenzen():
    """Long-Ratio muss zwischen 0 und 1 liegen."""
    cot = make_cot_df()
    spine = make_date_spine()
    result = compute_group_e(cot, spine)
    ratio = result["e_cot_long_ratio"].dropna()
    assert (ratio >= 0).all()
    assert (ratio <= 1).all()


# ── compute_group_f ────────────────────────────────────────────────────────

def test_group_f_schema():
    macro = pd.concat([
        make_macro_df("cpi_yoy"),
        make_macro_df("gdp_qoq"),
        make_macro_df("unemployment"),
    ])
    spine = make_date_spine()
    result = compute_group_f(macro, pd.DataFrame(), spine)
    expected = {
        "f_esi_cpi", "f_esi_gdp", "f_esi_unemployment",
        "f_esi_composite", "f_calendar_bias",
    }
    assert expected.issubset(set(result.columns))


def test_group_f_composite_ist_mittel_der_komponenten():
    """f_esi_composite muss dem Mittel der ESI-Komponenten entsprechen."""
    macro = pd.concat([
        make_macro_df("cpi_yoy"),
        make_macro_df("gdp_qoq"),
        make_macro_df("unemployment"),
    ])
    spine = make_date_spine()
    result = compute_group_f(macro, pd.DataFrame(), spine)
    # Pruefen fuer Zeilen, wo alle 3 Komponenten verfuegbar sind
    full_rows = result.dropna(subset=["f_esi_cpi", "f_esi_gdp", "f_esi_unemployment"])
    if full_rows.empty:
        pytest.skip("Nicht genug Daten fuer Composite-Test")
    manual_mean = full_rows[["f_esi_cpi", "f_esi_gdp", "f_esi_unemployment"]].mean(axis=1)
    diff = (full_rows["f_esi_composite"] - manual_mean).abs()
    assert (diff < 1e-9).all()


def test_group_f_esi_arbeitslosigkeit_invertiert():
    """Wenn Arbeitslosigkeit sinkt (negatives Trend-Delta), muss f_esi_unemployment steigen."""
    rows = []
    # Sinkende Arbeitslosigkeit (gut fuer Wirtschaft = positiver Surprise)
    for i, d in enumerate(pd.date_range("2018-01-01", "2020-06-30", freq="MS")):
        rows.append({"date": d, "value": 10.0 - i * 0.05, "currency": "USD",
                     "indicator": "unemployment", "series_id": "UNRATE"})
    macro = pd.DataFrame(rows)
    macro["date"] = pd.to_datetime(macro["date"])

    spine = make_date_spine()
    result = compute_group_f(macro, pd.DataFrame(), spine)
    esi_unemp = result.loc[
        (result["currency"] == "USD") & result["f_esi_unemployment"].notna(),
        "f_esi_unemployment"
    ]
    # Sinkende Arbeitslosigkeit sollte ueberwiegend positive ESI-Scores liefern
    assert esi_unemp.mean() > 0


def test_group_f_calendar_bias_aus_events():
    """f_calendar_bias muss nach Events aus dem Kalender gesetzt sein."""
    macro = make_macro_df("cpi_yoy")
    spine = make_date_spine()
    cal = make_calendar_df(currencies=["USD"])
    result = compute_group_f(macro, cal, spine)
    usd_bias = result.loc[
        (result["currency"] == "USD") & result["f_calendar_bias"].notna(),
        "f_calendar_bias"
    ]
    assert len(usd_bias) > 0


def test_group_f_esi_werte_zwischen_minus3_und_3():
    """ESI-Scores muessen auf [-3, 3] geclippt sein."""
    macro = pd.concat([
        make_macro_df("cpi_yoy"),
        make_macro_df("gdp_qoq"),
        make_macro_df("unemployment"),
    ])
    spine = make_date_spine()
    result = compute_group_f(macro, pd.DataFrame(), spine)
    for col in ["f_esi_cpi", "f_esi_gdp", "f_esi_unemployment"]:
        valid = result[col].dropna()
        assert (valid >= -3.0).all(), f"{col} unterschreitet -3"
        assert (valid <=  3.0).all(), f"{col} ueberschreitet +3"


# ── merge_all_features ────────────────────────────────────────────────────

def _make_all_groups(spine: pd.DataFrame):
    macro_ir  = make_macro_df("interest_rate")
    macro_cpi = make_macro_df("cpi_yoy")
    macro_gdp = make_macro_df("gdp_qoq")
    macro_ue  = make_macro_df("unemployment")
    macro_all = pd.concat([macro_ir, macro_cpi, macro_gdp, macro_ue])
    cot       = make_cot_df()
    cal       = make_calendar_df()

    group_a = compute_group_a(macro_all, cal, spine)
    group_b = compute_group_b(macro_all, cal, spine)
    group_e = compute_group_e(cot, spine)
    group_f = compute_group_f(macro_all, cal, spine)
    return group_a, group_b, group_e, group_f


def test_merge_schema():
    spine = make_date_spine()
    a, b, e, f = _make_all_groups(spine)
    result = merge_all_features(a, b, e, f)

    all_expected = {
        "date", "currency",
        "a_interest_rate", "a_rate_spread_vs_usd",
        "b_cpi_yoy", "b_cpi_vs_target",
        "e_cot_net_position", "e_cot_extreme_flag",
        "f_esi_composite", "f_calendar_bias",
    }
    assert all_expected.issubset(set(result.columns))


def test_merge_keine_doppelten_spalten():
    spine = make_date_spine()
    a, b, e, f = _make_all_groups(spine)
    result = merge_all_features(a, b, e, f)
    assert len(result.columns) == len(set(result.columns))


def test_merge_zeilen_gleich_date_spine():
    spine = make_date_spine()
    a, b, e, f = _make_all_groups(spine)
    result = merge_all_features(a, b, e, f)
    assert len(result) == len(spine)


def test_merge_alle_g7_waehrungen():
    spine = make_date_spine()
    a, b, e, f = _make_all_groups(spine)
    result = merge_all_features(a, b, e, f)
    assert set(result["currency"].unique()) == set(G7_CURRENCIES)


# ── save_to_parquet ────────────────────────────────────────────────────────

def test_save_parquet_erstellt_datei():
    spine = make_date_spine()
    a, b, e, f = _make_all_groups(spine)
    features = merge_all_features(a, b, e, f)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_to_parquet(features, Path(tmpdir))
        assert path.exists()
        assert path.suffix == ".parquet"
        assert "features_g7_" in path.name


def test_save_parquet_roundtrip():
    spine = make_date_spine()
    a, b, e, f = _make_all_groups(spine)
    features = merge_all_features(a, b, e, f)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_to_parquet(features, Path(tmpdir))
        loaded = pd.read_parquet(path)
        assert len(loaded) == len(features)
        assert set(loaded.columns) == set(features.columns)


# ── Integrations-Pipeline ─────────────────────────────────────────────────

def test_pipeline_mit_fehlenden_cot_daten():
    """Pipeline laeuft durch, auch wenn keine CoT-Daten vorhanden sind."""
    from src.data.feature_engineer import run as fe_run

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        macro_dir = tmp / "macro"
        macro_dir.mkdir()
        features_dir = tmp / "features"
        features_dir.mkdir()

        # Nur Zinsdaten speichern (kein CoT, kein Kalender)
        macro_ir = make_macro_df("interest_rate")
        macro_ir.to_parquet(macro_dir / "macro_interest_rate_20200630.parquet", index=False)

        result = fe_run(
            cot_dir=tmp / "cot",          # existiert nicht → graceful degradation
            macro_dir=macro_dir,
            calendar_dir=tmp / "calendar",  # existiert nicht → graceful degradation
            features_dir=features_dir,
            start=START,
            end=END,
        )

    assert not result.empty
    assert "date" in result.columns
    assert "currency" in result.columns
    # A-Features sollten vorhanden sein (Zinsdaten geladen)
    assert "a_interest_rate" in result.columns


def test_pipeline_vollstaendig():
    """Vollstaendige Pipeline mit allen Datenquellen."""
    from src.data.feature_engineer import run as fe_run

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        macro_dir = tmp / "macro"
        cot_dir   = tmp / "cot"
        cal_dir   = tmp / "calendar"
        feat_dir  = tmp / "features"
        for d in (macro_dir, cot_dir, cal_dir, feat_dir):
            d.mkdir()

        # Alle Makro-Indikatoren speichern
        for indicator in ["interest_rate", "cpi_yoy", "gdp_qoq", "unemployment"]:
            df = make_macro_df(indicator)
            df.to_parquet(macro_dir / f"macro_{indicator}_20200630.parquet", index=False)

        # CoT speichern
        cot = make_cot_df()
        cot.to_parquet(cot_dir / "cot_g7_20200630.parquet", index=False)

        # Kalender speichern
        cal = make_calendar_df()
        cal.to_parquet(cal_dir / "calendar_g7_20200630.parquet", index=False)

        result = fe_run(
            cot_dir=cot_dir,
            macro_dir=macro_dir,
            calendar_dir=cal_dir,
            features_dir=feat_dir,
            start=START,
            end=END,
        )

        # Schema pruefen (innerhalb des tempdir-Blocks, damit die Dateien noch existieren)
        assert not result.empty
        assert set(result["currency"].unique()) == set(G7_CURRENCIES)
        assert (result["date"].dt.weekday < 5).all()   # Nur Handelstage

        # Alle Feature-Gruppen muessen Spalten enthalten
        a_cols = [c for c in result.columns if c.startswith("a_")]
        b_cols = [c for c in result.columns if c.startswith("b_")]
        e_cols = [c for c in result.columns if c.startswith("e_")]
        f_cols = [c for c in result.columns if c.startswith("f_")]
        assert len(a_cols) >= 4
        assert len(b_cols) >= 4
        assert len(e_cols) >= 5
        assert len(f_cols) >= 4

        # Parquet-Datei muss erstellt worden sein
        parquet_files = list(feat_dir.glob("*.parquet"))
        assert len(parquet_files) == 1
