"""
Feature Engineering Modul — Forex ML Terminal.

Laedt CoT, Makro und Kalender-Daten und berechnet alle ML-Features
fuer die Modellgruppen A (Zinsen), B (Inflation), E (CoT), F (ESI).

Datenquellen:
  CoT      → data/raw/cot/cot_g7_*.parquet      (woechentlich)
  Makro    → data/raw/macro/macro_*_*.parquet    (monatlich/quartalsweise)
  Kalender → data/raw/calendar/calendar_g7_*.parquet (ereignisbasiert)

Ausgabe: data/features/features_g7_YYYYMMDD.parquet
Schema:  Eine Zeile pro Waehrung pro Handelstag.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Konstanten ─────────────────────────────────────────────────────────────

G7_CURRENCIES: list[str] = ["USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD"]

COT_DIR      = Path("data/raw/cot")
MACRO_DIR    = Path("data/raw/macro")
CALENDAR_DIR = Path("data/raw/calendar")
FEATURES_DIR = Path("data/features")

CB_INFLATION_TARGET = 2.0   # Ziel der meisten Zentralbanken

# Extremwert-Schwellen fuer CoT Kontraindikator (E4)
COT_EXTREME_HIGH = 90.0
COT_EXTREME_LOW  = 10.0

# Rolling-Fenster in Handelstagen
WINDOW_3Y   = 756    # ≈ 3 Jahre (252 × 3) — fuer historischen Zinsdurchschnitt
WINDOW_1Y   = 252    # ≈ 1 Jahr — fuer ESI-Baseline
WINDOW_3M   = 63     # ≈ 3 Monate — fuer CPI-Trend
WINDOW_6M   = 126    # ≈ 6 Monate — fuer CPI-Trend
WINDOW_4W   = 20     # ≈ 4 Wochen — fuer CoT-Trend
WINDOW_1W   = 5      # ≈ 1 Woche — fuer CoT-Flow-Beschleunigung

# Forward-Fill-Grenzen (Tage)
FF_LIMIT_MACRO = 45  # Monatsdaten: max 45 Tage vorwaerts fuellen
FF_LIMIT_COT   = 10  # Wochendaten: max 10 Tage vorwaerts fuellen

# Lookahead-Fenster fuer Zinserwartung aus Kalender
RATE_EVENT_LOOKAHEAD_DAYS = 14

# ESI: Bias-Fenster fuer Kalender-Events
ESI_BIAS_WINDOW = 90  # Tage

# ── Laden ──────────────────────────────────────────────────────────────────

def _load_latest(pattern: str, data_dir: Path) -> pd.DataFrame:
    """Laedt die neueste Parquet-Datei, die dem Pattern entspricht."""
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Keine Parquet-Datei fuer Pattern '{pattern}' in {data_dir}"
        )
    path = files[-1]
    logger.info("Lade: %s", path)
    return pd.read_parquet(path)


def load_cot(data_dir: Path = COT_DIR) -> pd.DataFrame:
    df = _load_latest("cot_g7_*.parquet", data_dir)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_macro(indicator: str, data_dir: Path = MACRO_DIR) -> pd.DataFrame:
    df = _load_latest(f"macro_{indicator}_*.parquet", data_dir)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_calendar(data_dir: Path = CALENDAR_DIR) -> pd.DataFrame:
    try:
        df = _load_latest("calendar_g7_*.parquet", data_dir)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        logger.warning("Keine Kalenderdaten gefunden — kalenderbasierte Features werden NaN")
        return pd.DataFrame(
            columns=["date", "currency", "event_name", "impact", "forecast", "previous"]
        )


# ── Hilfsfunktionen ────────────────────────────────────────────────────────

def _parse_numeric(s) -> float:
    """
    Konvertiert Kalender-Wertstrings zu float.

    Beispiele: '175K' → 175000, '4.25%' → 4.25, '1.5M' → 1500000, '' → NaN
    """
    if s is None:
        return float("nan")
    s = str(s).strip()
    if not s:
        return float("nan")
    s = s.replace(",", "").replace("%", "")
    multiplier = 1.0
    upper = s.upper()
    if upper.endswith("T"):
        multiplier, s = 1e12, s[:-1]
    elif upper.endswith("B"):
        multiplier, s = 1e9, s[:-1]
    elif upper.endswith("M"):
        multiplier, s = 1e6, s[:-1]
    elif upper.endswith("K"):
        multiplier, s = 1e3, s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return float("nan")


def _wide_to_long(wide: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Konvertiert Wide DataFrame (DatetimeIndex, Currency-Spalten) zu Long Format.

    Der Index muss "date" heissen.
    Gibt DataFrame mit Spalten [date, currency, col_name] zurueck.
    """
    g7_cols = [c for c in G7_CURRENCIES if c in wide.columns]
    long = (
        wide[g7_cols]
        .reset_index()
        .melt(id_vars="date", var_name="currency", value_name=col_name)
    )
    return long[["date", "currency", col_name]]


def _macro_to_daily(
    macro_long: pd.DataFrame,
    indicator: str,
    date_spine: pd.DataFrame,
    ff_limit: int = FF_LIMIT_MACRO,
) -> pd.DataFrame:
    """
    Pivotiert Langform-Makrodaten auf Tagesfrequenz und forward-filled Luecken.

    Rueckgabe: Wide DataFrame (DatetimeIndex "date", Spalten = Waehrungen).
    Bei fehlenden Daten wird ein leeres DataFrame mit NaN-Werten zurueckgegeben.
    """
    all_dates = pd.DatetimeIndex(sorted(date_spine["date"].unique()))

    subset = macro_long[macro_long["indicator"] == indicator][
        ["date", "currency", "value"]
    ].copy()

    if subset.empty:
        logger.warning("Keine Makrodaten fuer Indikator '%s'", indicator)
        empty = pd.DataFrame(np.nan, index=all_dates, columns=G7_CURRENCIES)
        empty.index.name = "date"
        return empty

    pivoted = subset.pivot_table(
        index="date", columns="currency", values="value", aggfunc="last"
    )
    daily = pivoted.reindex(all_dates).ffill(limit=ff_limit)
    daily.index.name = "date"
    return daily


def _cot_to_daily(
    cot_df: pd.DataFrame,
    feature: str,
    date_spine: pd.DataFrame,
    ff_limit: int = FF_LIMIT_COT,
) -> pd.DataFrame:
    """
    Pivotiert eine CoT-Feature-Spalte auf Tagesfrequenz und forward-filled.

    Rueckgabe: Wide DataFrame (DatetimeIndex "date", Spalten = Waehrungen).
    """
    all_dates = pd.DatetimeIndex(sorted(date_spine["date"].unique()))

    if feature not in cot_df.columns:
        empty = pd.DataFrame(np.nan, index=all_dates, columns=G7_CURRENCIES)
        empty.index.name = "date"
        return empty

    pivoted = cot_df.pivot_table(
        index="date", columns="currency", values=feature, aggfunc="last"
    )
    daily = pivoted.reindex(all_dates).ffill(limit=ff_limit)
    daily.index.name = "date"
    return daily


def build_date_spine(
    currencies: list[str] = G7_CURRENCIES,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Erstellt ein leeres (date × currency)-Raster fuer alle Handelstage.

    Standard: 2015-01-01 bis heute. Gibt DataFrame [date, currency] zurueck.
    """
    if start is None:
        start = "2015-01-01"
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    dates = pd.bdate_range(start=start, end=end, freq="B")
    idx = pd.MultiIndex.from_product(
        [dates, currencies], names=["date", "currency"]
    )
    return pd.DataFrame(index=idx).reset_index()


# ── Gruppe A: Zinspolitik ──────────────────────────────────────────────────

def compute_group_a(
    macro_long: pd.DataFrame,
    calendar_df: pd.DataFrame,
    date_spine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet Zinspolitik-Features (A1, A2, A3-Proxy) fuer alle G7-Waehrungen.

    a_interest_rate      — aktueller Leitzins (A1)
    a_rate_hist_avg_3y   — rollierender 3-Jahres-Durchschnitt (A1)
    a_rate_dev_from_avg  — Abweichung vom historischen Durchschnitt (A1)
    a_rate_spread_vs_usd — Zinsdifferenz gegenueber USD (A2)
    a_rate_expected_change — erwartete Zinsaenderung aus Kalender (A3-Proxy)
    """
    rates = _macro_to_daily(macro_long, "interest_rate", date_spine)

    hist_avg  = rates.rolling(WINDOW_3Y, min_periods=30).mean()
    dev       = rates - hist_avg

    usd_rates = rates["USD"] if "USD" in rates.columns else pd.Series(np.nan, index=rates.index)
    spread    = rates.sub(usd_rates, axis=0)

    result = date_spine.copy()
    for wide, name in [
        (rates,    "a_interest_rate"),
        (hist_avg, "a_rate_hist_avg_3y"),
        (dev,      "a_rate_dev_from_avg"),
        (spread,   "a_rate_spread_vs_usd"),
    ]:
        long_df = _wide_to_long(wide, name)
        result = result.merge(long_df, on=["date", "currency"], how="left")

    # A3-Proxy: Zinserwartung aus Kalenderevents (forecast - aktueller Kurs)
    result["a_rate_expected_change"] = np.nan

    if not calendar_df.empty and "event_name" in calendar_df.columns:
        rate_kw = r"interest rate|rate decision|monetary policy|base rate"
        rate_events = calendar_df[
            calendar_df["event_name"].str.lower().str.contains(
                rate_kw, na=False, regex=True
            )
        ].copy()
        rate_events["forecast_n"] = rate_events["forecast"].apply(_parse_numeric)
        rate_events = rate_events.dropna(subset=["forecast_n"])
        rate_events = rate_events[rate_events["forecast_n"].abs() > 0]

        for _, ev in rate_events.iterrows():
            ev_date = pd.Timestamp(ev["date"])
            currency = ev["currency"]
            forecast_n = ev["forecast_n"]

            before_ev = result[
                (result["currency"] == currency)
                & (result["date"] <= ev_date)
                & result["a_interest_rate"].notna()
            ]
            if before_ev.empty:
                continue

            current_rate = before_ev["a_interest_rate"].iloc[-1]
            expected_change = forecast_n - current_rate

            window_mask = (
                (result["currency"] == currency)
                & (result["date"] >= ev_date - pd.Timedelta(days=RATE_EVENT_LOOKAHEAD_DAYS))
                & (result["date"] <= ev_date)
            )
            result.loc[window_mask, "a_rate_expected_change"] = expected_change

    return result


# ── Gruppe B: Inflation ────────────────────────────────────────────────────

def compute_group_b(
    macro_long: pd.DataFrame,
    calendar_df: pd.DataFrame,
    date_spine: pd.DataFrame,
    inflation_target: float = CB_INFLATION_TARGET,
) -> pd.DataFrame:
    """
    Berechnet Inflations-Features (B1–B4) fuer alle G7-Waehrungen.

    b_cpi_yoy       — aktueller CPI YoY (B1)
    b_cpi_vs_target — CPI minus ZB-Ziel (B1)
    b_cpi_trend_3m  — CPI-Veraenderung ueber 3 Monate (B2)
    b_cpi_trend_6m  — CPI-Veraenderung ueber 6 Monate (B2)
    b_cpi_vs_usd    — CPI relativ zu USD (B4)
    b_cpi_surprise  — Abweichung actual/previous vs forecast aus Kalender (B3)
    """
    cpi = _macro_to_daily(macro_long, "cpi_yoy", date_spine)

    cpi_3m_ago = cpi.shift(WINDOW_3M)
    cpi_6m_ago = cpi.shift(WINDOW_6M)
    trend_3m   = cpi - cpi_3m_ago
    trend_6m   = cpi - cpi_6m_ago

    usd_cpi = cpi["USD"] if "USD" in cpi.columns else pd.Series(np.nan, index=cpi.index)
    vs_usd  = cpi.sub(usd_cpi, axis=0)

    result = date_spine.copy()
    for wide, name in [
        (cpi,      "b_cpi_yoy"),
        (trend_3m, "b_cpi_trend_3m"),
        (trend_6m, "b_cpi_trend_6m"),
        (vs_usd,   "b_cpi_vs_usd"),
    ]:
        long_df = _wide_to_long(wide, name)
        result = result.merge(long_df, on=["date", "currency"], how="left")

    # B1: Abstand vom ZB-Inflationsziel
    result["b_cpi_vs_target"] = result["b_cpi_yoy"] - inflation_target

    # B3: CPI Surprise — actual (oder previous als Proxy) vs forecast
    result["b_cpi_surprise"] = np.nan

    if not calendar_df.empty and "event_name" in calendar_df.columns:
        cpi_events = calendar_df[
            calendar_df["event_name"].str.lower().str.contains(
                r"cpi|consumer price|inflation", na=False, regex=True
            )
        ].copy()
        cpi_events["forecast_n"] = cpi_events["forecast"].apply(_parse_numeric)

        has_actual = "actual" in cpi_events.columns
        if has_actual:
            cpi_events["ref_n"] = cpi_events["actual"].apply(_parse_numeric)
        else:
            cpi_events["ref_n"] = cpi_events["previous"].apply(_parse_numeric)

        cpi_events["surprise"] = cpi_events["ref_n"] - cpi_events["forecast_n"]
        cpi_events = cpi_events.dropna(subset=["surprise"])
        cpi_events["date"] = pd.to_datetime(cpi_events["date"])
        cpi_events = cpi_events.sort_values("date")

        for currency in G7_CURRENCIES:
            curr_ev = cpi_events[cpi_events["currency"] == currency]
            if curr_ev.empty:
                continue
            mask = result["currency"] == currency
            curr_dates = result.loc[mask, "date"]

            surprise_series = pd.Series(
                curr_ev["surprise"].values,
                index=pd.DatetimeIndex(curr_ev["date"].values),
                dtype="float64",
            )
            # Events koennen auf Wochenenden/Feiertagen liegen → dichten Index
            # aufbauen, der beide Datumsreihen abdeckt, dann reindex auf Handelstage
            all_dense = pd.DatetimeIndex(
                sorted(set(list(curr_dates.values) + list(curr_ev["date"].values)))
            )
            dense = surprise_series.reindex(all_dense).ffill(limit=90)
            reindexed = dense.reindex(curr_dates.values)
            result.loc[mask, "b_cpi_surprise"] = reindexed.values

    return result


# ── Gruppe E: CoT & Positionierung ────────────────────────────────────────

def compute_group_e(
    cot_df: pd.DataFrame,
    date_spine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet CoT-Features (E1–E4) fuer alle G7-Waehrungen.

    e_cot_net_position    — Non-Comm Long minus Short (E1)
    e_cot_trend_4w        — Veraenderung der Net Position ueber 4 Wochen (E1)
    e_cot_long_ratio      — Long-Anteil an Long+Short (E1)
    e_cot_net_percentile  — 3-Jahres-Perzentil der Net Position (E2)
    e_cot_weekly_change   — Wochenveraenderung der Net Position (E3)
    e_cot_flow_acceleration — Beschleunigung des wochentlichen Flows (E3)
    e_cot_extreme_flag    — Kontraindikator: +1 bei >90%, -1 bei <10% (E4)
    """
    net_pos   = _cot_to_daily(cot_df, "net_position",   date_spine)
    pct       = _cot_to_daily(cot_df, "net_percentile", date_spine)
    ratio     = _cot_to_daily(cot_df, "long_ratio",     date_spine)
    net_chg   = _cot_to_daily(cot_df, "net_change",     date_spine)

    trend_4w  = net_pos - net_pos.shift(WINDOW_4W)
    accel     = net_chg - net_chg.shift(WINDOW_1W)

    result = date_spine.copy()
    for wide, name in [
        (net_pos,  "e_cot_net_position"),
        (trend_4w, "e_cot_trend_4w"),
        (ratio,    "e_cot_long_ratio"),
        (pct,      "e_cot_net_percentile"),
        (net_chg,  "e_cot_weekly_change"),
        (accel,    "e_cot_flow_acceleration"),
    ]:
        long_df = _wide_to_long(wide, name)
        result = result.merge(long_df, on=["date", "currency"], how="left")

    # E4: Extremwert-Flag — NaN wenn kein Perzentil verfuegbar
    pct_col = result["e_cot_net_percentile"]
    result["e_cot_extreme_flag"] = np.where(
        pct_col.isna(),   np.nan,
        np.where(pct_col > COT_EXTREME_HIGH,  1.0,
        np.where(pct_col < COT_EXTREME_LOW,  -1.0, 0.0)),
    )

    return result


# ── Gruppe F: Economic Surprise Index ─────────────────────────────────────

def compute_group_f(
    macro_long: pd.DataFrame,
    calendar_df: pd.DataFrame,
    date_spine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet Economic Surprise Index Features (F2) fuer alle G7-Waehrungen.

    Makro-basierter ESI: Wie stark weicht der aktuelle Wert vom 1-Jahres-Trend ab?
    (Rolling z-score, Arbeitslosigkeit invertiert)

    f_esi_cpi           — Inflations-Surprise-Score (z-score)
    f_esi_gdp           — BIP-Surprise-Score (z-score)
    f_esi_unemployment  — Arbeitsmarkt-Surprise-Score (z-score, invertiert)
    f_esi_composite     — Gleichgewichtetes Mittel aller verfuegbaren Komponenten
    f_calendar_bias     — Rollierende Erwartungsabweichung aus Kalender (sign)
    """
    result = date_spine.copy()
    esi_cols: list[str] = []

    for indicator, col_name, invert in [
        ("cpi_yoy",     "f_esi_cpi",         False),
        ("gdp_qoq",     "f_esi_gdp",         False),
        ("unemployment","f_esi_unemployment", True),
    ]:
        wide = _macro_to_daily(macro_long, indicator, date_spine)

        roll_mean = wide.rolling(WINDOW_1Y, min_periods=30).mean()
        roll_std  = wide.rolling(WINDOW_1Y, min_periods=30).std()
        z_score   = (wide - roll_mean) / (roll_std + 1e-9)

        if invert:
            z_score = -z_score

        z_score = z_score.clip(-3.0, 3.0)

        long_df = _wide_to_long(z_score, col_name)
        result = result.merge(long_df, on=["date", "currency"], how="left")
        esi_cols.append(col_name)

    # Composite ESI: Mittel aller verfuegbaren Komponenten
    if esi_cols:
        result["f_esi_composite"] = result[esi_cols].mean(axis=1, skipna=True)
    else:
        result["f_esi_composite"] = np.nan

    # Kalender-Erwartungsbias: sign(forecast - previous) rollierend gemittelt
    result["f_calendar_bias"] = np.nan

    if not calendar_df.empty and "event_name" in calendar_df.columns:
        cal = calendar_df.copy()
        cal["date"] = pd.to_datetime(cal["date"])
        cal["forecast_n"] = cal["forecast"].apply(_parse_numeric)
        cal["previous_n"] = cal["previous"].apply(_parse_numeric)
        cal = cal.dropna(subset=["forecast_n", "previous_n"])
        cal["bias_sign"] = np.sign(cal["forecast_n"] - cal["previous_n"])
        cal = cal.sort_values("date")

        for currency in G7_CURRENCIES:
            curr_ev = cal[cal["currency"] == currency]
            if curr_ev.empty:
                continue
            mask = result["currency"] == currency
            curr_dates = result.loc[mask, "date"]

            bias_series = pd.Series(
                curr_ev["bias_sign"].values,
                index=pd.DatetimeIndex(curr_ev["date"].values),
                dtype="float64",
            )
            # Reindex → forward-fill → rolling mean ueber ESI_BIAS_WINDOW Tage
            daily_bias = bias_series.reindex(curr_dates.values).ffill(limit=ESI_BIAS_WINDOW)
            rolling_bias = daily_bias.rolling(window=ESI_BIAS_WINDOW, min_periods=1).mean()
            result.loc[mask, "f_calendar_bias"] = rolling_bias.values

    return result


# ── Zusammenfuehren & Speichern ────────────────────────────────────────────

def merge_all_features(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    group_e: pd.DataFrame,
    group_f: pd.DataFrame,
) -> pd.DataFrame:
    """Fuegt alle Feature-Gruppen zu einem breiten DataFrame zusammen."""
    keys = ["date", "currency"]

    a_cols = [c for c in group_a.columns if c.startswith("a_")]
    b_cols = [c for c in group_b.columns if c.startswith("b_")]
    e_cols = [c for c in group_e.columns if c.startswith("e_")]
    f_cols = [c for c in group_f.columns if c.startswith("f_")]

    result = group_a[keys + a_cols].copy()
    result = result.merge(group_b[keys + b_cols], on=keys, how="left")
    result = result.merge(group_e[keys + e_cols], on=keys, how="left")
    result = result.merge(group_f[keys + f_cols], on=keys, how="left")

    return result.sort_values(keys).reset_index(drop=True)


def save_to_parquet(df: pd.DataFrame, out_dir: Path = FEATURES_DIR) -> Path:
    """Speichert die Feature-Tabelle als Parquet-Datei."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"features_g7_{today}.parquet"
    df.to_parquet(path, index=False)
    logger.info(
        "Features gespeichert: %d Zeilen × %d Spalten → %s",
        len(df), len(df.columns), path,
    )
    return path


# ── Pipeline ───────────────────────────────────────────────────────────────

def run(
    cot_dir: Path = COT_DIR,
    macro_dir: Path = MACRO_DIR,
    calendar_dir: Path = CALENDAR_DIR,
    features_dir: Path = FEATURES_DIR,
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Vollstaendige Feature-Engineering-Pipeline.

    Laedt alle Rohdaten, berechnet Features fuer Gruppen A, B, E, F
    und speichert als Parquet in features_dir.
    """
    logger.info("=== Feature Engineering Pipeline startet ===")

    # CoT laden
    try:
        cot_df = load_cot(cot_dir)
        logger.info("CoT: %d Zeilen, %d Waehrungen", len(cot_df), cot_df["currency"].nunique())
        if not cot_df.empty and pd.Timestamp(start) < cot_df["date"].min():
            start = cot_df["date"].min().strftime("%Y-%m-%d")
    except FileNotFoundError as exc:
        logger.error("CoT-Daten fehlen: %s", exc)
        empty_cols = ["date", "currency", "net_position", "net_percentile",
                      "net_change", "long_ratio", "non_comm_long", "non_comm_short"]
        cot_df = pd.DataFrame(columns=empty_cols)

    # Makro laden (alle 4 Indikatoren zusammen in ein langes DataFrame)
    macro_frames: list[pd.DataFrame] = []
    for indicator in ["interest_rate", "cpi_yoy", "gdp_qoq", "unemployment"]:
        try:
            df = load_macro(indicator, macro_dir)
            macro_frames.append(df)
            logger.info("Makro '%s': %d Zeilen", indicator, len(df))
        except FileNotFoundError as exc:
            logger.warning("Makrodaten fehlen fuer '%s': %s", indicator, exc)

    macro_long = (
        pd.concat(macro_frames, ignore_index=True)
        if macro_frames
        else pd.DataFrame(columns=["date", "currency", "indicator", "value"])
    )

    # Kalender laden
    calendar_df = load_calendar(calendar_dir)

    # Datums-Raster aufbauen
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("Date-Spine: %s bis %s", start, end)
    date_spine = build_date_spine(start=start, end=end)

    # Features berechnen
    logger.info("Berechne Gruppe A (Zinspolitik) ...")
    group_a = compute_group_a(macro_long, calendar_df, date_spine)

    logger.info("Berechne Gruppe B (Inflation) ...")
    group_b = compute_group_b(macro_long, calendar_df, date_spine)

    logger.info("Berechne Gruppe E (CoT-Positionierung) ...")
    group_e = compute_group_e(cot_df, date_spine)

    logger.info("Berechne Gruppe F (Economic Surprise Index) ...")
    group_f = compute_group_f(macro_long, calendar_df, date_spine)

    features = merge_all_features(group_a, group_b, group_e, group_f)
    logger.info(
        "Feature-Tabelle: %d Zeilen × %d Spalten (%d Waehrungen)",
        len(features), len(features.columns), features["currency"].nunique(),
    )

    save_to_parquet(features, features_dir)
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run()
    feature_cols = [c for c in result.columns if c not in ("date", "currency")]
    print(f"\n=== Feature-Schema ({len(feature_cols)} Features) ===")
    for col in feature_cols:
        non_null = result[col].notna().sum()
        print(f"  {col:40s}  non-null={non_null:>8d}")
    print(f"\nLetzte Zeile pro Waehrung:")
    print(result.groupby("currency").last()[feature_cols[:8]].to_string())
