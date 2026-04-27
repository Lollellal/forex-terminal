"""
FRED (Federal Reserve Economic Data) Makrodaten-Fetcher.

Laedt G7-Makrodaten ueber die kostenlose FRED API:
  - Leitzinsen (monatlich)
  - CPI YoY Inflation (monatlich)
  - Arbeitslosenquote (monatlich)
  - BIP QoQ Wachstum (quartalsweise)

API Key benoetigt: https://fred.stlouisfed.org/docs/api/api_key.html
In .env eintragen: FRED_API_KEY=<dein_key>
Output: Parquet in data/raw/macro/
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

# ── Konstanten ─────────────────────────────────────────────────────────────

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
DATA_DIR = Path("data/raw/macro")

# Historische Daten ab 2000 laden
OBSERVATION_START = "2000-01-01"

# FRED Series IDs: Zentralbank-Leitzinsen (monatlich)
INTEREST_RATE_SERIES: dict[str, str] = {
    "USD": "FEDFUNDS",           # Fed Funds Rate
    "EUR": "ECBDFR",             # ECB Deposit Facility Rate
    "GBP": "BOERUKM",            # Bank of England Official Bank Rate
    "JPY": "IRSTCB01JPM156N",    # Bank of Japan Policy Rate (OECD)
    "CAD": "IRSTCB01CAM156N",    # Bank of Canada Overnight Rate (OECD)
    "CHF": "IRSTCB01CHM156N",    # SNB Policy Rate (OECD)
    "AUD": "IRSTCB01AUM156N",    # RBA Cash Rate (OECD)
}

# FRED Series IDs: CPI YoY Inflation (monatlich, % Vorjahresveraenderung)
CPI_YOY_SERIES: dict[str, str] = {
    "USD": "CPALTT01USM659N",    # US CPI All Items YoY
    "EUR": "CPALTT01EZM659N",    # Euro Area HICP YoY
    "GBP": "CPALTT01GBM659N",    # UK CPI YoY
    "JPY": "CPALTT01JPM659N",    # Japan CPI YoY
    "CAD": "CPALTT01CAM659N",    # Canada CPI YoY
    "CHF": "CPALTT01CHM659N",    # Switzerland CPI YoY
    "AUD": "CPALTT01AUM659N",    # Australia CPI YoY
}

# FRED Series IDs: Arbeitslosenquote (monatlich, %)
UNEMPLOYMENT_SERIES: dict[str, str] = {
    "USD": "UNRATE",             # US Unemployment Rate
    "EUR": "LRHUTTTTEZM156S",    # Euro Area Unemployment Rate (OECD)
    "GBP": "LRHUTTTTGBM156S",    # UK Unemployment Rate (OECD)
    "JPY": "LRHUTTTTJPM156S",    # Japan Unemployment Rate (OECD)
    "CAD": "LRHUTTTTCAM156S",    # Canada Unemployment Rate (OECD)
    "CHF": "LRHUTTTTCHM156S",    # Switzerland Unemployment Rate (OECD)
    "AUD": "LRHUTTTTAUM156S",    # Australia Unemployment Rate (OECD)
}

# FRED Series IDs: BIP QoQ Wachstum (quartalsweise, %)
# Hinweis: USD-Serie (SAAR) ist annualisiert; OECD-Serien sind echte QoQ-Raten.
GDP_QOQ_SERIES: dict[str, str] = {
    "USD": "A191RL1Q225SBEA",    # US Real GDP Percent Change (SAAR)
    "EUR": "NAEXKP01EZQ657S",    # Euro Area GDP QoQ (OECD)
    "GBP": "NAEXKP01GBQ657S",    # UK GDP QoQ (OECD)
    "JPY": "NAEXKP01JPQ657S",    # Japan GDP QoQ (OECD)
    "CAD": "NAEXKP01CAQ657S",    # Canada GDP QoQ (OECD)
    "CHF": "NAEXKP01CHQ657S",    # Switzerland GDP QoQ (OECD)
    "AUD": "NAEXKP01AUQ657S",    # Australia GDP QoQ (OECD)
}

ALL_INDICATORS: dict[str, dict[str, str]] = {
    "interest_rate": INTEREST_RATE_SERIES,
    "cpi_yoy": CPI_YOY_SERIES,
    "unemployment": UNEMPLOYMENT_SERIES,
    "gdp_qoq": GDP_QOQ_SERIES,
}

G7_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD"}


# ── API Key ────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Liest FRED API Key aus Umgebungsvariablen."""
    key = os.getenv("FRED_API_KEY", "")
    if not key or key == "your_fred_api_key_here":
        raise EnvironmentError(
            "FRED_API_KEY fehlt oder nicht gesetzt.\n"
            "Kostenlos registrieren: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Dann in .env eintragen: FRED_API_KEY=<dein_key>"
        )
    return key


# ── FRED API Aufruf ────────────────────────────────────────────────────────

def fetch_series(
    series_id: str,
    api_key: str,
    observation_start: str = OBSERVATION_START,
) -> pd.Series:
    """
    Ruft eine einzelne FRED-Zeitreihe ab.

    FRED kodiert fehlende Werte als "." — diese werden zu None konvertiert.
    Gibt eine pandas Series mit DatetimeIndex zurueck.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    payload = resp.json()
    if "observations" not in payload:
        raise ValueError(
            f"Unerwartetes FRED-Antwortformat fuer '{series_id}'. "
            f"Schluesse: {list(payload.keys())}"
        )

    obs = payload["observations"]
    dates = pd.to_datetime([o["date"] for o in obs])
    values = [None if o["value"] == "." else float(o["value"]) for o in obs]

    series = pd.Series(values, index=dates, name=series_id, dtype="float64")

    if len(series) > 0:
        logger.info(
            "  %s: %d Punkte (%s bis %s)",
            series_id, len(series),
            series.index.min().date(),
            series.index.max().date(),
        )
    else:
        logger.warning("  %s: Keine Datenpunkte erhalten", series_id)

    return series


# ── Indikator-Fetcher ──────────────────────────────────────────────────────

def fetch_indicator(
    indicator_name: str,
    series_map: dict[str, str],
    api_key: str,
    observation_start: str = OBSERVATION_START,
) -> pd.DataFrame:
    """
    Laedt alle G7-Serien fuer einen Makro-Indikator.

    Schlaegt eine einzelne Waehrung fehl, wird sie uebersprungen (mit Warnung).
    Scheitern alle Waehrungen, wird RuntimeError geworfen.

    Rueckgabe: langes DataFrame mit Spalten [date, currency, indicator, series_id, value].
    """
    logger.info("Lade Indikator: %s", indicator_name)
    frames: list[pd.DataFrame] = []

    for currency, series_id in series_map.items():
        try:
            series = fetch_series(series_id, api_key, observation_start)
            df_curr = series.reset_index()
            df_curr.columns = ["date", "value"]
            df_curr["currency"] = currency
            df_curr["indicator"] = indicator_name
            df_curr["series_id"] = series_id
            frames.append(df_curr)
        except Exception as exc:
            logger.warning(
                "  FEHLER %s (%s): %s — Waehrung wird uebersprungen.",
                currency, series_id, exc,
            )

    if not frames:
        raise RuntimeError(
            f"Keine Daten fuer Indikator '{indicator_name}' geladen. "
            "Alle Serien sind fehlgeschlagen."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["value"])
    df = df.sort_values(["currency", "date"]).reset_index(drop=True)

    logger.info(
        "%s: %d Zeilen, %d Waehrungen",
        indicator_name, len(df), df["currency"].nunique(),
    )
    return df


# ── Speichern ──────────────────────────────────────────────────────────────

def save_to_parquet(
    df: pd.DataFrame,
    indicator: str,
    out_dir: Path = DATA_DIR,
) -> Path:
    """Speichert Indikator-DataFrame als Parquet-Datei."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime("%Y%m%d")
    path = out_dir / f"macro_{indicator}_{today}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Gespeichert: %d Zeilen → %s", len(df), path)
    return path


# ── Laden (Hilfsfunktion fuer spaetere Schritte) ──────────────────────────

def load_latest(indicator: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Laedt die neueste Parquet-Datei fuer einen Indikator."""
    pattern = f"macro_{indicator}_*.parquet"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Keine Parquet-Dateien fuer '{indicator}' in {data_dir}. "
            "Bitte zuerst run() ausfuehren."
        )
    path = files[-1]
    logger.info("Lade: %s", path)
    return pd.read_parquet(path)


# ── Pipeline ───────────────────────────────────────────────────────────────

def run(
    out_dir: Path = DATA_DIR,
    observation_start: str = OBSERVATION_START,
) -> dict[str, pd.DataFrame]:
    """Vollstaendige Pipeline: Alle 4 Indikatoren laden und als Parquet speichern."""
    api_key = _get_api_key()
    results: dict[str, pd.DataFrame] = {}

    for indicator_name, series_map in ALL_INDICATORS.items():
        df = fetch_indicator(indicator_name, series_map, api_key, observation_start)
        save_to_parquet(df, indicator_name, out_dir)
        results[indicator_name] = df

    total_rows = sum(len(df) for df in results.values())
    logger.info("Pipeline abgeschlossen. Gesamt: %d Zeilen in %d Indikatoren.", total_rows, len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run()

    for indicator, df in results.items():
        latest = (
            df.groupby("currency")
            .last()[["date", "value"]]
            .rename(columns={"value": indicator})
        )
        print(f"\n=== Letzte {indicator}-Werte ===")
        print(latest.to_string())
