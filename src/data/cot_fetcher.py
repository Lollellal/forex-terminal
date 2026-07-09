"""
CFTC Commitments of Traders (CoT) — Legacy Futures Only fetcher.

Laedt woechentliche Commercial- und Small-Speculator-Positionierung fuer G7-Waehrungen,
berechnet Net Positions, Wochenveraenderungen und 52-Wochen-Perzentile (COT-Index).
Output: Parquet in data/raw/cot/

Strategie:
  Commercial (Hedger): Folge-Signal — hohes Perzentil = bullish
  Small Spec (Non-Reportable): Kontra-Signal — hohes Perzentil = bearish

Datenquelle: CFTC per-Jahr-Archive
  https://www.cftc.gov/files/dea/history/deahistfoYYYY.zip
"""

import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Konstanten ─────────────────────────────────────────────────────────────

# Per-Jahr-Archive (Legacy Futures Only, TXT-Format)
CFTC_BASE_URL  = "https://www.cftc.gov/files/dea/history/deahistfo{year}.zip"
CFTC_START_YEAR = 2010  # frühestes Jahr für Gruppe-A Features (3-Jahres-Fenster)

DATA_DIR = Path("data/raw/cot")

# COT-Index Fenster: 52 Wochen (1 Jahr) — Standard fuer COT-Index-Strategie
COT_INDEX_WINDOW = 52

# G7-Waehrungen: Substring-Match gegen CFTC-Marktnamen (Grossbuchstaben)
# "EURO FX - CHICAGO MERCANTILE EXCHANGE"    → EUR
# "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE" → GBP
# "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE"  → JPY  usw.
# DXY: "U.S. DOLLAR INDEX - ICE FUTURES U.S." → USD (Korb-Signal für alle Paare)
G7_CURRENCY_FILTERS: dict[str, str] = {
    "EUR": "EURO FX - CHICAGO",
    "GBP": "BRITISH POUND - CHICAGO",
    "JPY": "JAPANESE YEN - CHICAGO",
    "CAD": "CANADIAN DOLLAR - CHICAGO",
    "CHF": "SWISS FRANC - CHICAGO",
    "AUD": "AUSTRALIAN DOLLAR - CHICAGO",
    "NZD": "NEW ZEALAND DOLLAR - CHICAGO",
    "USD": "USD INDEX - ICE FUTURES",
}

# Spalten-Mapping: CFTC Legacy TXT → interner Name
# (seit ~2017 nutzt CFTC Leerzeichen-Stil statt Underscore-Stil)
COLUMN_MAP: dict[str, str] = {
    "As of Date in Form YYYY-MM-DD":           "date",
    # Commercial (Hedger) — Folge-Signal
    "Commercial Positions-Long (All)":          "comm_long",
    "Commercial Positions-Short (All)":         "comm_short",
    "Change in Commercial-Long (All)":          "comm_change_long",
    "Change in Commercial-Short (All)":         "comm_change_short",
    # Non-Reportable / Small Speculator — Kontra-Signal
    "Nonreportable Positions-Long (All)":       "small_long",
    "Nonreportable Positions-Short (All)":      "small_short",
    "Change in Nonreportable-Long (All)":       "small_change_long",
    "Change in Nonreportable-Short (All)":      "small_change_short",
    "Open Interest (All)":                      "open_interest",
}

# Marktname-Spalte in Legacy TXT
MARKET_COL = "Market and Exchange Names"


# ── Download ─────────────────────────────���─────────────────────────────────

def _download_year(year: int) -> pd.DataFrame:
    """Laedt ein einzelnes Jahres-Archiv und gibt rohen DataFrame zurueck."""
    url = CFTC_BASE_URL.format(year=year)
    logger.debug("  CoT %d: %s", year, url)
    resp = requests.get(url, timeout=120)
    if resp.status_code == 404:
        logger.debug("  CoT %d: nicht gefunden (404)", year)
        return pd.DataFrame()
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        txt_files = [f for f in zf.namelist() if f.lower().endswith(".txt")]
        if not txt_files:
            raise ValueError(f"Keine .txt-Datei im {year}-Archiv. Inhalt: {zf.namelist()}")
        with zf.open(txt_files[0]) as fh:
            df = pd.read_csv(fh, low_memory=False)

    logger.debug("  CoT %d: %d Zeilen", year, len(df))
    return df


def download_raw_cot(
    start_year: int = CFTC_START_YEAR,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Laedt CFTC Legacy Futures Daten fuer mehrere Jahre und kombiniert sie.

    start_year: frühestes Jahr (Standard 2010, brauchen 3 Jahre extra fuer Perzentil)
    end_year:   letztes Jahr (Standard: aktuelles Kalenderjahr)
    """
    if end_year is None:
        end_year = datetime.now(timezone.utc).year

    years = list(range(start_year, end_year + 1))
    logger.info("Lade CoT-Daten fuer %d Jahre parallel ...", len(years))

    year_frames: dict[int, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_download_year, y): y for y in years}
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    year_frames[year] = df
            except Exception as exc:
                logger.warning("CoT %d: Download fehlgeschlagen: %s", year, exc)

    if not year_frames:
        raise RuntimeError(
            f"Keine CoT-Daten geladen (Jahre {start_year}–{end_year}). "
            "CFTC-URL geaendert? Bitte cot_fetcher.py pruefen."
        )

    frames = [year_frames[y] for y in sorted(year_frames)]
    combined = pd.concat(frames, ignore_index=True)
    logger.info("CoT Rohdaten: %d Zeilen aus %d Jahren", len(combined), len(frames))
    return combined


# ── Filtern & Extrahieren ────────────────────────────���─────────────────────

def _detect_currency(market_name: str) -> str | None:
    """Gibt Waehrungscode zurueck wenn Marktname zu G7 passt."""
    upper = market_name.upper()
    for currency, pattern in G7_CURRENCY_FILTERS.items():
        if pattern in upper:
            return currency
    return None


def filter_g7_currencies(df: pd.DataFrame) -> pd.DataFrame:
    """Filtert auf G7-Waehrungs-Futures und fuegt 'currency'-Spalte hinzu."""
    if MARKET_COL not in df.columns:
        raise KeyError(
            f"Spalte '{MARKET_COL}' nicht gefunden. "
            f"Vorhandene Spalten (erste 10): {list(df.columns[:10])}"
        )

    df = df.copy()
    df["currency"] = df[MARKET_COL].apply(_detect_currency)
    result = df[df["currency"].notna()].copy()

    found   = sorted(result["currency"].unique())
    missing = sorted(set(G7_CURRENCY_FILTERS) - set(found))
    logger.info("Gefundene Waehrungen: %s", found)
    if missing:
        logger.warning("Fehlende Waehrungen (kein direkter Futures-Markt): %s", missing)

    return result


def extract_cot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrahiert und berechnet Commercial + Small Spec CoT-Features."""
    missing_cols = [c for c in COLUMN_MAP if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Fehlende Spalten: {missing_cols}. "
            f"Vorhandene Spalten: {list(df.columns)}"
        )

    result = df[["currency"] + list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP).copy()
    result["date"] = pd.to_datetime(result["date"])

    # CFTC-Dateien enthalten manchmal String-Dtype — explizit zu float konvertieren
    num_cols = [
        "comm_long", "comm_short", "comm_change_long", "comm_change_short",
        "small_long", "small_short", "small_change_long", "small_change_short",
        "open_interest",
    ]
    for col in num_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    # Commercial: Net Position + wöchentliche Veränderung
    result["comm_net"]    = result["comm_long"]  - result["comm_short"]
    result["comm_change"] = result["comm_change_long"] - result["comm_change_short"]

    # Small Speculator (Non-Reportable): Net Position + wöchentliche Veränderung
    result["small_net"]    = result["small_long"]  - result["small_short"]
    result["small_change"] = result["small_change_long"] - result["small_change_short"]

    return result.sort_values(["currency", "date"]).reset_index(drop=True)


# ── Perzentil-Berechnung ────────────────────────��──────────────────────────

def _rolling_index(series: pd.Series, window: int) -> pd.Series:
    """
    COT-Index: Wo steht der aktuelle Wert im N-Wochen-Fenster?
    0 = historisches Minimum, 100 = historisches Maximum.
    Formel: (current - min) / (max - min) * 100
    Erste Datenpunkte (< min_periods) = 50.0 (neutral).
    """
    roll_min = series.rolling(window, min_periods=4).min()
    roll_max = series.rolling(window, min_periods=4).max()
    rng = roll_max - roll_min
    idx = (series - roll_min) / rng.where(rng > 0, other=np.nan) * 100
    return idx.fillna(50.0)


def calculate_cot_indices(df: pd.DataFrame, window: int = COT_INDEX_WINDOW) -> pd.DataFrame:
    """
    Berechnet COT-Index (0–100) fuer Commercial und Small Spec pro Waehrung.

    comm_index:  0 = Commercials so short wie nie, 100 = so long wie nie → Folge-Signal
    small_index: 0 = Small Spec so short wie nie, 100 = so long wie nie → Kontra-Signal
    """
    df = df.copy()

    def _apply_index(group: pd.DataFrame, col: str) -> pd.Series:
        return _rolling_index(group[col], window)

    df["comm_index"]  = df.groupby("currency", group_keys=False).apply(
        lambda g: _apply_index(g, "comm_net")
    )
    df["small_index"] = df.groupby("currency", group_keys=False).apply(
        lambda g: _apply_index(g, "small_net")
    )
    return df


# ── Speichern ────────────────────────────────────��────────────────────────��

def save_to_parquet(df: pd.DataFrame, out_dir: Path = DATA_DIR) -> Path:
    """Speichert verarbeitete CoT-Daten als Parquet-Datei."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path  = out_dir / f"cot_g7_{today}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Gespeichert: %d Zeilen → %s", len(df), path)
    return path


# ── Pipeline ─────────────────────────────────���─────────────────────────────

def run(
    start_year: int = CFTC_START_YEAR,
    out_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Vollstaendige Pipeline: Download → Filter → Extraktion → COT-Index → Parquet."""
    logger.info("Lade CFTC CoT-Daten (%d–heute) ...", start_year)
    raw      = download_raw_cot(start_year=start_year)
    g7       = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    final    = calculate_cot_indices(features)
    save_to_parquet(final, out_dir)
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run()
    summary = (
        result.groupby("currency")
        .last()[["date", "comm_net", "comm_index", "small_net", "small_index"]]
        .sort_values("currency")
    )
    print("\n=== Letzte CoT-Werte pro Waehrung ===")
    print(summary.to_string())
