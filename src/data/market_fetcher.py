"""
FRED Market Data Fetcher — VIX, Equities, Commodities, Yields.

Laedt Marktdaten fuer Gruppen G (Risikoumfeld) und H (Rohstoffe)
sowie 10-Jahres-Renditen fuer Gruppe F (Yield Spread):

  Global:  VIX, S&P 500, Gold, WTI Crude Oil, Copper, GDPNow, Weekly Economic Index,
           St. Louis Fed Financial Stress Index
  Yields:  US 2Y / 10Y Treasury + G7 10-Jahres-Staatsanleihen
           (fuer f_yield_spread_vs_usd in Gruppe F4)

API Key: FRED_API_KEY in .env (gleicher Key wie macro_fetcher)
Output:  data/raw/market/market_data_YYYYMMDD.parquet
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FRED_BASE_URL   = "https://api.stlouisfed.org/fred/series/observations"
DATA_DIR        = Path("data/raw/market")
OBSERVATION_START = "2000-01-01"

# ── Serien-Definitionen ────────────────────────────────────────────────────

# Globale Marktserien (taeglich, nicht waehrungs-spezifisch)
GLOBAL_SERIES: dict[str, str] = {
    "VIX":    "VIXCLS",             # CBOE Volatility Index
    "SP500":  "SP500",              # S&P 500 Index
    "GOLD":   "GOLDAMGBD228NLBM",   # Gold London Fix (USD/troy oz) — bekannt: von FRED entfernt (Mai 2025), liefert NaN
    "OIL":    "DCOILWTICO",         # WTI Crude Oil (USD/barrel)
    "COPPER": "PCOPPUSDM",          # Global Copper Price (monatlich)
    "US_2Y":  "DGS2",               # US 2-Year Treasury Yield
    "US_10Y": "DGS10",              # US 10-Year Treasury Yield
    "WEI":    "WEI",                # NY/Dallas Fed Weekly Economic Index
    "STLFSI": "STLFSI4",            # St. Louis Fed Financial Stress Index (woechentlich)
    # GDPNOW bewusst NICHT aufgenommen: der Standard-FRED-Endpoint liefert nur den
    # zuletzt revidierten Wert pro Quartal, nicht die tatsaechliche taegliche
    # Nowcast-Historie (verifiziert 2026-07-09 ueber ALFRED-Vintages). Damit waere
    # das Feature lookahead-belastet. Reaktivierung nur mit ALFRED-Vintage-Fetch.
}

# G7 10-Jahres-Staatsanleihen-Renditen (fuer F4 Yield Spread)
G7_10Y_YIELD_SERIES: dict[str, str] = {
    "USD": "DGS10",             # US 10Y (taeglich)
    "EUR": "IRLTLT01EZM156N",   # Euro Area 10Y (monatlich)
    "GBP": "IRLTLT01GBM156N",   # UK 10Y (monatlich)
    "JPY": "IRLTLT01JPM156N",   # Japan 10Y (monatlich)
    "CAD": "IRLTLT01CAM156N",   # Canada 10Y (monatlich)
    "CHF": "IRLTLT01CHM156N",   # Switzerland 10Y (monatlich)
    "AUD": "IRLTLT01AUM156N",   # Australia 10Y (monatlich)
    "NZD": "IRLTLT01NZM156N",   # New Zealand 10Y (monatlich, OECD)
}


# ── API ────────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.getenv("FRED_API_KEY", "")
    if not key or key == "your_fred_api_key_here":
        raise EnvironmentError(
            "FRED_API_KEY fehlt. In .env eintragen: FRED_API_KEY=<key>"
        )
    return key


def _fetch_series(series_id: str, api_key: str, start: str = OBSERVATION_START) -> pd.Series:
    """Ruft eine FRED-Zeitreihe ab und gibt eine DatetimeIndex-Series zurueck."""
    params = {
        "series_id":        series_id,
        "api_key":          api_key,
        "file_type":        "json",
        "observation_start": start,
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    dates  = pd.to_datetime([o["date"] for o in obs])
    values = [float("nan") if o["value"] == "." else float(o["value"]) for o in obs]
    s = pd.Series(values, index=dates, name=series_id, dtype="float64")
    logger.info("  %s: %d Punkte", series_id, s.notna().sum())
    return s


# ── Haupt-Fetch ────────────────────────────────────────────────────────────

def fetch_market_data(
    api_key: str | None = None,
    start: str = OBSERVATION_START,
) -> pd.DataFrame:
    """
    Laedt alle globalen Marktserien + G7 10Y Yields von FRED.

    Rueckgabe: Wide DataFrame (DatetimeIndex "date"), Spalten:
      VIX, SP500, GOLD, OIL, COPPER, US_2Y, US_10Y,
      YIELD_USD, YIELD_EUR, YIELD_GBP, YIELD_JPY, YIELD_CAD, YIELD_CHF, YIELD_AUD
    """
    if api_key is None:
        api_key = _get_api_key()

    frames: dict[str, pd.Series] = {}

    logger.info("Lade globale Marktserien ...")
    for col, sid in GLOBAL_SERIES.items():
        try:
            frames[col] = _fetch_series(sid, api_key, start)
        except Exception as exc:
            logger.warning("  FAIL %s (%s): %s", col, sid, exc)

    logger.info("Lade G7 10-Jahres-Renditen ...")
    for currency, sid in G7_10Y_YIELD_SERIES.items():
        col = f"YIELD_{currency}"
        try:
            frames[col] = _fetch_series(sid, api_key, start)
        except Exception as exc:
            logger.warning("  FAIL %s (%s): %s", col, sid, exc)

    if not frames:
        raise RuntimeError("Keine Marktdaten geladen — alle FRED-Abfragen fehlgeschlagen.")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    logger.info(
        "Marktdaten: %d Zeitpunkte, %d Spalten (%s–%s)",
        len(df), len(df.columns),
        df.index.min().date(), df.index.max().date(),
    )
    return df


# ── Speichern ──────────────────────────────────────────────────────────────

def save_to_parquet(df: pd.DataFrame, out_dir: Path = DATA_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime("%Y%m%d")
    path = out_dir / f"market_data_{today}.parquet"
    df.reset_index().to_parquet(path, index=False)
    logger.info("Gespeichert: %d Zeilen → %s", len(df), path)
    return path


def load_latest(data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    """Laedt die neueste Marktdaten-Parquet, gibt None zurueck wenn nicht vorhanden."""
    files = sorted(data_dir.glob("market_data_*.parquet"))
    if not files:
        logger.warning("Keine Marktdaten gefunden in %s", data_dir)
        return None
    df = pd.read_parquet(files[-1])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.info("Marktdaten geladen: %s (%d Zeilen)", files[-1].name, len(df))
    return df


# ── Pipeline ───────────────────────────────────────────────────────────────

def run(out_dir: Path = DATA_DIR, start: str = OBSERVATION_START) -> pd.DataFrame:
    """Vollstaendige Pipeline: Laden und Speichern aller Marktdaten."""
    api_key = _get_api_key()
    df = fetch_market_data(api_key=api_key, start=start)
    save_to_parquet(df, out_dir)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = run()
    print(f"\n=== Marktdaten-Schema ({len(df.columns)} Spalten) ===")
    for col in df.columns:
        n = df[col].notna().sum()
        last = df[col].dropna().iloc[-1] if n > 0 else "n/a"
        print(f"  {col:15} non-null={n:>7}  letzter={last:.2f}" if isinstance(last, float) else f"  {col:15} non-null={n:>7}")
