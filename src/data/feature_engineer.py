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

G7_CURRENCIES: list[str] = ["USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "NZD"]

COT_DIR      = Path("data/raw/cot")
MACRO_DIR    = Path("data/raw/macro")
CALENDAR_DIR = Path("data/raw/calendar")
FEATURES_DIR = Path("data/features")
MARKET_DIR   = Path("data/raw/market")

CB_INFLATION_TARGET = 2.0   # Ziel der meisten Zentralbanken

# COT-Index Schwellen fuer E4 Combined Signal
COT_SIGNAL_HIGH = 80.0   # >80 = klares Signal (bullish Commercial / bearish Small Spec)
COT_SIGNAL_LOW  = 20.0   # <20 = klares Gegensignal

# Rolling-Fenster in Handelstagen
WINDOW_3Y   = 756    # ≈ 3 Jahre (252 × 3) — fuer historischen Zinsdurchschnitt
WINDOW_1Y   = 252    # ≈ 1 Jahr — fuer ESI-Baseline
WINDOW_3M   = 63     # ≈ 3 Monate — fuer CPI-Trend / 1 Quartal GDP
WINDOW_6M   = 126    # ≈ 6 Monate — fuer CPI-Trend / 2 Quartale GDP
WINDOW_4W   = 20     # ≈ 4 Wochen — fuer CoT-Trend
WINDOW_1W   = 5      # ≈ 1 Woche — fuer CoT-Flow-Beschleunigung

FF_LIMIT_GDP = 90    # Quartalsdata: max 90 Tage vorwaerts fuellen

# Forward-Fill-Grenzen (Tage)
FF_LIMIT_MACRO = 45  # Monatsdaten: max 45 Tage vorwaerts fuellen
FF_LIMIT_COT   = 10  # Wochendaten: max 10 Tage vorwaerts fuellen

# Lookahead-Fenster fuer Zinserwartung aus Kalender
RATE_EVENT_LOOKAHEAD_DAYS = 14

# ESI: Bias-Fenster fuer Kalender-Events
ESI_BIAS_WINDOW = 90  # Tage

# Rolling-Fenster fuer Markt-Features (Handelstage)
WINDOW_1M  = 21   # ≈ 1 Monat
WINDOW_3M_DAYS = 63  # ≈ 3 Monate (Alias fuer Gruppen G/H)

# Publikationslag-Naeherungen (Kalendertage) je Makro-Indikator. FRED liefert den
# Datenpunkt mit dem Referenzzeitraum-Datum (z.B. GDP Q1 am 2015-01-01), NICHT mit dem
# tatsaechlichen Veroeffentlichungsdatum. Grobe, literaturuebliche Naeherungen -- kein
# exakter Vintage-Kalender verfuegbar. Siehe MACRO_ML_SYSTEM_AUDIT.md Abschnitt 1.1 und
# der isolierte Test in der ehemaligen feature_engineer_macrofix.py-Forschungskopie.
MACRO_PUBLICATION_LAG_DAYS: dict[str, int] = {
    "cpi_yoy":       45,   # CPI YoY, monatlich
    "unemployment":  35,   # Arbeitslosigkeit, monatlich (NZD quartalsweise, Naeherung bleibt gleich)
    "gdp_qoq":       120,  # GDP QoQ, quartalsweise
    "interest_rate": 32,   # Zinsen, meist Monatsdurchschnitts-Serien
}

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


def load_market(data_dir: Path = MARKET_DIR) -> "pd.DataFrame | None":
    """Laedt neueste Marktdaten-Parquet (VIX, Oil, Gold …). None wenn nicht vorhanden."""
    files = sorted(data_dir.glob("market_data_*.parquet"))
    if not files:
        logger.warning("Keine Marktdaten in %s — G/H Features werden NaN", data_dir)
        return None
    df = pd.read_parquet(files[-1])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.info("Marktdaten geladen: %s (%d Zeilen, %d Spalten)", files[-1].name, len(df), len(df.columns))
    return df


def load_calendar(data_dir: Path = CALENDAR_DIR) -> pd.DataFrame:
    """
    Laedt und konkateniert ALLE monatlichen Backfill-Parquets (calendar_g7_YYYYMM.parquet,
    6-stelliges Muster). Frueher wurde per _load_latest() nur die eine lexikographisch letzte
    Datei geladen -- bei monatlich partitionierten Dateien blieb dadurch der komplette
    historische Backfill wirkungslos (nur der juengste Monat wurde je gesehen).
    """
    files = sorted(data_dir.glob("calendar_g7_??????.parquet"))
    if not files:
        logger.warning("Keine Kalenderdaten gefunden — kalenderbasierte Features werden NaN")
        return pd.DataFrame(
            columns=["date", "currency", "event_name", "impact", "actual", "forecast", "previous"]
        )

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "time", "currency", "event_name"])
    logger.info(
        "Kalender geladen: %d Dateien, %d Events (dedupliziert), Zeitraum %s–%s",
        len(files), len(df), df["date"].min().date(), df["date"].max().date(),
    )
    return df


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
    reset = wide[g7_cols].reset_index()
    date_col = reset.columns[0]  # erster Eintrag nach reset_index() ist immer der Index
    long = reset.melt(id_vars=date_col, var_name="currency", value_name=col_name)
    long = long.rename(columns={date_col: "date"})
    return long[["date", "currency", col_name]]


def _macro_to_daily(
    macro_long: pd.DataFrame,
    indicator: str,
    date_spine: pd.DataFrame,
    ff_limit: int = FF_LIMIT_MACRO,
) -> pd.DataFrame:
    """
    Pivotiert Langform-Makrodaten auf Tagesfrequenz und forward-filled Luecken.

    Verschiebt das Referenzzeitraum-Datum vor dem Pivotieren um den indikator-spezifischen
    Publikationslag (MACRO_PUBLICATION_LAG_DAYS), damit kein Look-Ahead-Bias entsteht --
    analog zum COT-Muster in `_cot_to_daily()` (dort Handelstage, hier Kalendertage).

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

    lag_days = MACRO_PUBLICATION_LAG_DAYS.get(indicator)
    if lag_days is not None:
        subset["date"] = subset["date"] + pd.Timedelta(days=lag_days)
    else:
        logger.warning(
            "Kein Publikationslag fuer Indikator '%s' definiert -- "
            "Datum bleibt unveraendert (Lookahead-Risiko bestehen)", indicator
        )

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

    # CFTC "As of Date" ist Dienstag, aber Veröffentlichung erst Freitag (+3 Werktage).
    # Wir verschieben das Datum um 3 Werktage damit kein Look-Ahead entsteht.
    cot_shifted = cot_df.copy()
    cot_shifted["date"] = cot_shifted["date"] + pd.offsets.BusinessDay(3)

    pivoted = cot_shifted.pivot_table(
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

            # Mehrere CPI-Events am selben Tag (z.B. CPI YoY + Core CPI MoM) →
            # auf einen Wert pro Tag aggregieren, sonst doppelter Index beim Reindex
            daily_surprise = curr_ev.groupby("date")["surprise"].mean()

            surprise_series = pd.Series(
                daily_surprise.values,
                index=pd.DatetimeIndex(daily_surprise.index),
                dtype="float64",
            )
            # Events koennen auf Wochenenden/Feiertagen liegen → dichten Index
            # aufbauen, der beide Datumsreihen abdeckt, dann reindex auf Handelstage
            all_dense = pd.DatetimeIndex(
                sorted(set(list(curr_dates.values) + list(surprise_series.index.values)))
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
    Berechnet CoT-Features (E1–E3) fuer alle G7-Waehrungen.

    Commercial (Hedger) — Folge-Signal:
      e_cot_comm_net      — Commercial Net Position (E1)
      e_cot_comm_change   — Woechentliche Veraenderung (E1)
      e_cot_comm_index    — COT-Index 0-100 im 52W-Fenster (E1, E3)
      e_cot_comm_trend_4w — 4-Wochen-Momentum der Net Position (E1)

    Small Speculator / Non-Reportable — Kontra-Signal:
      e_cot_small_net      — Small Spec Net Position (E2)
      e_cot_small_change   — Woechentliche Veraenderung (E2)
      e_cot_small_index    — COT-Index 0-100 im 52W-Fenster (E2, E3)
      e_cot_small_trend_4w — 4-Wochen-Momentum der Net Position (E2)
    """
    comm_net   = _cot_to_daily(cot_df, "comm_net",    date_spine)
    comm_chg   = _cot_to_daily(cot_df, "comm_change", date_spine)
    comm_idx   = _cot_to_daily(cot_df, "comm_index",  date_spine)
    small_net  = _cot_to_daily(cot_df, "small_net",   date_spine)
    small_chg  = _cot_to_daily(cot_df, "small_change",date_spine)
    small_idx  = _cot_to_daily(cot_df, "small_index", date_spine)

    comm_trend_4w  = comm_net  - comm_net.shift(WINDOW_4W)
    small_trend_4w = small_net - small_net.shift(WINDOW_4W)
    divergence     = comm_idx  - small_idx   # E3-Signal: >0 bullish, <0 bearish

    # E4: DXY (USD Index) — Korb-Signal für alle Währungspaare.
    # DXY comm_index niedrig = Commercials short USD = USD bearish = Fremdwährungen bullish.
    # Das Signal ist identisch für alle Paare → USD-Spalte auf alle Währungen broadcasten.
    dxy_cot = cot_df[cot_df["currency"] == "USD"].copy()
    if not dxy_cot.empty:
        dxy_spine = build_date_spine(currencies=["USD"], start=str(date_spine["date"].min().date()))
        dxy_comm_idx  = _cot_to_daily(dxy_cot, "comm_index",  dxy_spine)["USD"]
        dxy_small_idx = _cot_to_daily(dxy_cot, "small_index", dxy_spine)["USD"]
        dxy_div       = dxy_comm_idx - dxy_small_idx
        # Auf alle Währungen broadcasten (jede Währungszeile bekommt denselben DXY-Wert)
        all_dates = comm_idx.index
        dxy_comm_wide  = pd.DataFrame(
            {c: dxy_comm_idx.reindex(all_dates)  for c in G7_CURRENCIES if c != "USD"}
        )
        dxy_small_wide = pd.DataFrame(
            {c: dxy_small_idx.reindex(all_dates) for c in G7_CURRENCIES if c != "USD"}
        )
        dxy_div_wide   = pd.DataFrame(
            {c: dxy_div.reindex(all_dates)       for c in G7_CURRENCIES if c != "USD"}
        )
    else:
        empty = pd.DataFrame(np.nan, index=comm_idx.index,
                             columns=[c for c in G7_CURRENCIES if c != "USD"])
        dxy_comm_wide = dxy_small_wide = dxy_div_wide = empty

    result = date_spine.copy()
    for wide, name in [
        (comm_net,       "e_cot_comm_net"),
        (comm_chg,       "e_cot_comm_change"),
        (comm_idx,       "e_cot_comm_index"),
        (comm_trend_4w,  "e_cot_comm_trend_4w"),
        (small_net,      "e_cot_small_net"),
        (small_chg,      "e_cot_small_change"),
        (small_idx,      "e_cot_small_index"),
        (small_trend_4w, "e_cot_small_trend_4w"),
        (divergence,     "e_cot_divergence"),
        (dxy_comm_wide,  "e_cot_dxy_comm_index"),
        (dxy_small_wide, "e_cot_dxy_small_index"),
        (dxy_div_wide,   "e_cot_dxy_divergence"),
    ]:
        result = result.merge(_wide_to_long(wide, name), on=["date", "currency"], how="left")

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

            # Mehrere Events am selben Tag → auf einen Wert pro Tag aggregieren,
            # sonst doppelter Index beim Reindex
            daily_bias_sign = curr_ev.groupby("date")["bias_sign"].mean()

            bias_series = pd.Series(
                daily_bias_sign.values,
                index=pd.DatetimeIndex(daily_bias_sign.index),
                dtype="float64",
            )
            # Reindex → forward-fill → rolling mean ueber ESI_BIAS_WINDOW Tage
            daily_bias = bias_series.reindex(curr_dates.values).ffill(limit=ESI_BIAS_WINDOW)
            rolling_bias = daily_bias.rolling(window=ESI_BIAS_WINDOW, min_periods=1).mean()
            result.loc[mask, "f_calendar_bias"] = rolling_bias.values

    return result


# ── Gruppe F: Yield Spread (F4) ───────────────────────────────────────────

def compute_group_f_yields(
    market_df: "pd.DataFrame | None",
    date_spine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet Anleihe-Yield-Features fuer Gruppe F (F4) aus Marktdaten.

    f_yield_10y          — 10-Jahres-Rendite dieser Waehrung
    f_yield_spread_vs_usd — Rendite-Spread vs. US-10Y (Kapitalfluss-Signal)
    """
    result = date_spine.copy()

    if market_df is None:
        result["f_yield_10y"]          = float("nan")
        result["f_yield_spread_vs_usd"] = float("nan")
        return result

    all_dates = pd.DatetimeIndex(sorted(date_spine["date"].unique()))

    # Referenz: US 10Y (taeglich)
    usd_10y_raw = market_df.get("YIELD_USD", pd.Series(dtype=float))
    usd_10y = usd_10y_raw.reindex(all_dates).ffill(limit=45)

    yield_frames: dict[str, pd.Series] = {}
    for currency in G7_CURRENCIES:
        col = f"YIELD_{currency}"
        raw = market_df.get(col, pd.Series(dtype=float))
        yield_frames[currency] = raw.reindex(all_dates).ffill(limit=45)

    wide_yield  = pd.DataFrame(yield_frames, index=all_dates)
    wide_spread = wide_yield.sub(usd_10y, axis=0)

    for wide, name in [(wide_yield, "f_yield_10y"), (wide_spread, "f_yield_spread_vs_usd")]:
        long_df = _wide_to_long(wide, name)
        result = result.merge(long_df, on=["date", "currency"], how="left")

    return result


# ── Gruppe G: Risikoumfeld ─────────────────────────────────────────────────

def _reindex_ffill(
    market_df: "pd.DataFrame",
    name: str,
    dates: pd.DatetimeIndex,
    limit: int = 5,
) -> pd.Series:
    """Reindexiert eine Marktdaten-Serie auf einen DatetimeIndex mit forward-fill."""
    raw = market_df.get(name, pd.Series(dtype=float))
    return raw.reindex(dates).ffill(limit=limit)


def _merge_global_features(
    result: pd.DataFrame,
    series_map: list[tuple["pd.Series", str]],
) -> pd.DataFrame:
    """Fuegt mehrere datums-indexierte Serien in einem einzigen Merge hinzu."""
    global_df = pd.DataFrame(
        {col: s for s, col in series_map},
    )
    global_df.index.name = "date"
    return result.merge(global_df.reset_index(), on="date", how="left")


def compute_group_g(
    market_df: "pd.DataFrame | None",
    date_spine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet Risikoumfeld-Features (G1–G4) fuer alle G7-Waehrungen.

    g_vix_level       — CBOE VIX Niveau (G1)
    g_vix_trend_1m    — VIX-Veraenderung ueber 1 Monat (G1)
    g_sp500_return_1m — S&P 500 1-Monats-Rendite (G2)
    g_sp500_return_3m — S&P 500 3-Monats-Rendite (G2)
    g_gold_return_1m  — Gold 1-Monats-Rendite (G3)
    g_gold_level      — Gold-Preis absolut (G3)
    g_yield_curve_us  — US 10Y minus 2Y Spread (G4)
    """
    result = date_spine.copy()
    g_cols = [
        "g_vix_level", "g_vix_trend_1m",
        "g_sp500_return_1m", "g_sp500_return_3m",
        "g_gold_return_1m", "g_gold_level", "g_yield_curve_us",
    ]
    if market_df is None:
        for col in g_cols:
            result[col] = float("nan")
        return result

    all_dates = pd.DatetimeIndex(sorted(date_spine["date"].unique()))
    vix   = _reindex_ffill(market_df, "VIX",   all_dates)
    sp500 = _reindex_ffill(market_df, "SP500", all_dates)
    gold  = _reindex_ffill(market_df, "GOLD",  all_dates)
    us10y = _reindex_ffill(market_df, "US_10Y", all_dates)
    us2y  = _reindex_ffill(market_df, "US_2Y",  all_dates)

    return _merge_global_features(result, [
        (vix,                               "g_vix_level"),
        (vix - vix.shift(WINDOW_1M),        "g_vix_trend_1m"),
        (sp500.pct_change(WINDOW_1M)    * 100, "g_sp500_return_1m"),
        (sp500.pct_change(WINDOW_3M_DAYS) * 100, "g_sp500_return_3m"),
        (gold.pct_change(WINDOW_1M)     * 100, "g_gold_return_1m"),
        (gold,                               "g_gold_level"),
        (us10y - us2y,                       "g_yield_curve_us"),
    ])


# ── Gruppe H: Rohstoffe ────────────────────────────────────────────────────

def compute_group_h(
    market_df: "pd.DataFrame | None",
    date_spine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet Rohstoff-Features (H1–H3) fuer alle G7-Waehrungen.

    h_oil_level        — WTI Roh-Oel-Preis absolut (H1)
    h_oil_return_1m    — WTI 1-Monats-Rendite in % (H1)
    h_oil_return_3m    — WTI 3-Monats-Rendite in % (H1)
    h_gold_level       — Gold-Preis absolut (H3)
    h_gold_return_1m   — Gold 1-Monats-Rendite in % (H3)
    h_copper_level     — Kupfer-Preis absolut (H2)
    h_copper_return_1m — Kupfer 1-Monats-Rendite in % (H2)
    """
    result = date_spine.copy()
    h_cols = [
        "h_oil_level", "h_oil_return_1m", "h_oil_return_3m",
        "h_gold_level", "h_gold_return_1m",
        "h_copper_level", "h_copper_return_1m",
    ]
    if market_df is None:
        for col in h_cols:
            result[col] = float("nan")
        return result

    all_dates = pd.DatetimeIndex(sorted(date_spine["date"].unique()))
    oil    = _reindex_ffill(market_df, "OIL",    all_dates)
    gold   = _reindex_ffill(market_df, "GOLD",   all_dates)
    copper = _reindex_ffill(market_df, "COPPER", all_dates, limit=45)

    return _merge_global_features(result, [
        (oil,                               "h_oil_level"),
        (oil.pct_change(WINDOW_1M)    * 100, "h_oil_return_1m"),
        (oil.pct_change(WINDOW_3M_DAYS) * 100, "h_oil_return_3m"),
        (gold,                              "h_gold_level"),
        (gold.pct_change(WINDOW_1M)   * 100, "h_gold_return_1m"),
        (copper,                            "h_copper_level"),
        (copper.pct_change(WINDOW_1M) * 100, "h_copper_return_1m"),
    ])


# ── Gruppe C: Arbeitsmarkt ─────────────────────────────────────────────────

def compute_group_c(
    macro_long: pd.DataFrame,
    date_spine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet Arbeitsmarkt-Features (C1–C4) fuer alle G7-Waehrungen.

    c_unemployment_rate  — aktuelle Arbeitslosenquote (C1)
    c_unemp_hist_avg_3y  — rollierender 3-Jahres-Durchschnitt (C1)
    c_unemp_dev_from_avg — Abweichung vom hist. Schnitt (negativ = enger Markt = bullish)
    c_unemp_vs_usd       — Differenz zur US-Arbeitslosenquote (C3)
    c_unemp_trend_3m     — 3-Monats-Veraenderung (negativ = Verbesserung = bullish)
    c_unemp_trend_6m     — 6-Monats-Veraenderung (C2)
    """
    unemp    = _macro_to_daily(macro_long, "unemployment", date_spine)
    hist_avg = unemp.rolling(WINDOW_3Y, min_periods=30).mean()
    dev      = unemp - hist_avg
    usd_u    = unemp["USD"] if "USD" in unemp.columns else unemp.mean(axis=1)
    vs_usd   = unemp.sub(usd_u, axis=0)
    trend_3m = unemp - unemp.shift(WINDOW_3M)
    trend_6m = unemp - unemp.shift(WINDOW_6M)

    result = date_spine.copy()
    for wide, name in [
        (unemp,    "c_unemployment_rate"),
        (hist_avg, "c_unemp_hist_avg_3y"),
        (dev,      "c_unemp_dev_from_avg"),
        (vs_usd,   "c_unemp_vs_usd"),
        (trend_3m, "c_unemp_trend_3m"),
        (trend_6m, "c_unemp_trend_6m"),
    ]:
        result = result.merge(_wide_to_long(wide, name), on=["date", "currency"], how="left")

    return result


# ── Gruppe D: Wachstum & PMI ───────────────────────────────────────────────

def compute_group_d(
    macro_long: pd.DataFrame,
    date_spine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet Wachstums-Features (D1–D4) fuer alle G7-Waehrungen.

    d_gdp_qoq           — BIP QoQ Wachstum (quartalsweise, forward-filled 90d)
    d_gdp_hist_avg_3y   — rollierender 3-Jahres-Durchschnitt (D1)
    d_gdp_dev_from_avg  — Abweichung vom hist. Schnitt (positiv = ueber Trend = bullish)
    d_gdp_vs_usd        — Wachstumsdifferenzial vs. USD (D3)
    d_gdp_trend_2q      — 2-Quartals-Momentum: GDP[t] - GDP[t-2Q] (D2)
    d_gdp_accel         — Beschleunigung: Differenz der 1-Quartals-Veraenderungen (D4)
    """
    gdp      = _macro_to_daily(macro_long, "gdp_qoq", date_spine, ff_limit=FF_LIMIT_GDP)
    hist_avg = gdp.rolling(WINDOW_3Y, min_periods=8).mean()
    dev      = gdp - hist_avg
    usd_g    = gdp["USD"] if "USD" in gdp.columns else gdp.mean(axis=1)
    vs_usd   = gdp.sub(usd_g, axis=0)
    trend_2q = gdp - gdp.shift(WINDOW_6M)       # 2 Quartale ≈ 126 Handelstage
    chg_1q   = gdp - gdp.shift(WINDOW_3M)       # 1 Quartal ≈ 63 Handelstage
    accel    = chg_1q - chg_1q.shift(WINDOW_3M)

    result = date_spine.copy()
    for wide, name in [
        (gdp,      "d_gdp_qoq"),
        (hist_avg, "d_gdp_hist_avg_3y"),
        (dev,      "d_gdp_dev_from_avg"),
        (vs_usd,   "d_gdp_vs_usd"),
        (trend_2q, "d_gdp_trend_2q"),
        (accel,    "d_gdp_accel"),
    ]:
        result = result.merge(_wide_to_long(wide, name), on=["date", "currency"], how="left")

    return result


# ── Zusammenfuehren & Speichern ────────────────────────────────────────────

def merge_all_features(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    group_c: pd.DataFrame | None = None,
    group_d: pd.DataFrame | None = None,
    group_e: pd.DataFrame | None = None,
    group_f: pd.DataFrame | None = None,
    group_f_yields: pd.DataFrame | None = None,
    group_g: pd.DataFrame | None = None,
    group_h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fuegt alle Feature-Gruppen zu einem breiten DataFrame zusammen."""
    keys = ["date", "currency"]

    def _select(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = [c for c in df.columns if c.startswith(prefix)]
        return df[keys + cols]

    result = _select(group_a, "a_").copy()
    result = result.merge(_select(group_b, "b_"), on=keys, how="left")

    if group_c is not None:
        result = result.merge(_select(group_c, "c_"), on=keys, how="left")

    if group_d is not None:
        result = result.merge(_select(group_d, "d_"), on=keys, how="left")

    if group_e is not None:
        result = result.merge(_select(group_e, "e_"), on=keys, how="left")

    if group_f is not None:
        result = result.merge(_select(group_f, "f_"), on=keys, how="left")

    if group_f_yields is not None:
        fy_cols = [c for c in group_f_yields.columns if c.startswith("f_")]
        result = result.merge(group_f_yields[keys + fy_cols], on=keys, how="left")

    if group_g is not None:
        result = result.merge(_select(group_g, "g_"), on=keys, how="left")

    if group_h is not None:
        result = result.merge(_select(group_h, "h_"), on=keys, how="left")

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
    market_dir: Path = MARKET_DIR,
    features_dir: Path = FEATURES_DIR,
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Vollstaendige Feature-Engineering-Pipeline.

    Laedt alle Rohdaten, berechnet Features fuer Gruppen A, B, E, F, G, H
    und speichert als Parquet in features_dir.
    Marktdaten (G, H) sind optional — fehlende Daten erzeugen NaN-Features.
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

    # Makro laden
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

    # Marktdaten laden (optional)
    market_df = load_market(market_dir)

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

    logger.info("Berechne Gruppe C (Arbeitsmarkt) ...")
    group_c = compute_group_c(macro_long, date_spine)

    logger.info("Berechne Gruppe D (Wachstum & PMI) ...")
    group_d = compute_group_d(macro_long, date_spine)

    logger.info("Berechne Gruppe E (CoT-Positionierung) ...")
    group_e = compute_group_e(cot_df, date_spine)

    logger.info("Berechne Gruppe F (Economic Surprise Index) ...")
    group_f = compute_group_f(macro_long, calendar_df, date_spine)

    logger.info("Berechne Gruppe F-Yields (Anleihe-Spreads) ...")
    group_f_yields = compute_group_f_yields(market_df, date_spine)

    logger.info("Berechne Gruppe G (Risikoumfeld) ...")
    group_g = compute_group_g(market_df, date_spine)

    logger.info("Berechne Gruppe H (Rohstoffe) ...")
    group_h = compute_group_h(market_df, date_spine)

    features = merge_all_features(
        group_a, group_b, group_c, group_d, group_e, group_f,
        group_f_yields, group_g, group_h,
    )
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
