"""
CFTC Commitments of Traders (CoT) — Legacy Futures Only fetcher.

Laedt woechentliche Non-Commercial Positionierung fuer G7-Waehrungen,
berechnet Net Position, Wochenveraenderung und 3-Jahres-Perzentil.
Output: Parquet in data/raw/cot/
"""

import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Konstanten ─────────────────────────────────────────────────────────────

# Kombinierte Historiendatei (Legacy Futures Only, alle Jahre)
CFTC_URL = "https://www.cftc.gov/dea/newcot/deafuturesonly.zip"

DATA_DIR = Path("data/raw/cot")

# 3 Jahre * 52 Wochen = 156 Datenpunkte fuer Perzentil-Fenster
PERCENTILE_WINDOW = 156

# G7-Waehrungen: Substring-Match gegen CFTC-Marktnamen (Grossbuchstaben)
G7_CURRENCY_FILTERS: dict[str, str] = {
    "EUR": "EURO FX",
    "GBP": "BRITISH POUND STERLING",
    "JPY": "JAPANESE YEN",
    "CAD": "CANADIAN DOLLAR",
    "CHF": "SWISS FRANC",
    "AUD": "AUSTRALIAN DOLLAR",
    "USD": "U.S. DOLLAR INDEX",
}

# Spalten-Mapping: CFTC-Name → interner Name
COLUMN_MAP: dict[str, str] = {
    "Report_Date_as_YYYY-MM-DD": "date",
    "NonComm_Positions_Long_All": "non_comm_long",
    "NonComm_Positions_Short_All": "non_comm_short",
    "Change_in_NonComm_Long_All": "change_long",
    "Change_in_NonComm_Short_All": "change_short",
    "Open_Interest_All": "open_interest",
}

# ── Download ───────────────────────────────────────────────────────────────

def download_raw_cot(url: str = CFTC_URL) -> pd.DataFrame:
    """Laedt die CFTC ZIP-Datei und gibt rohe DataFrame zurueck."""
    logger.info("Lade CoT-Daten von %s ...", url)
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        txt_files = [f for f in zf.namelist() if f.lower().endswith(".txt")]
        if not txt_files:
            raise ValueError(
                f"Keine .txt-Datei in ZIP gefunden. Inhalt: {zf.namelist()}"
            )
        target = txt_files[0]
        logger.info("Lese '%s' aus ZIP ...", target)
        with zf.open(target) as fh:
            df = pd.read_csv(fh, low_memory=False)

    logger.info("Roh-Daten: %d Zeilen, %d Spalten", len(df), len(df.columns))
    return df


# ── Filtern & Extrahieren ──────────────────────────────────────────────────

def _detect_currency(market_name: str) -> str | None:
    """Gibt Waehrungscode zurueck wenn Marktname zu G7 passt."""
    upper = market_name.upper()
    for currency, pattern in G7_CURRENCY_FILTERS.items():
        if pattern in upper:
            return currency
    return None


def filter_g7_currencies(df: pd.DataFrame) -> pd.DataFrame:
    """Filtert auf G7-Waehrungs-Futures und fuegt 'currency'-Spalte hinzu."""
    name_col = "Market_and_Exchange_Names"
    if name_col not in df.columns:
        raise KeyError(
            f"Spalte '{name_col}' nicht gefunden. "
            f"Vorhandene Spalten (erste 10): {list(df.columns[:10])}"
        )

    df = df.copy()
    df["currency"] = df[name_col].apply(_detect_currency)
    result = df[df["currency"].notna()].copy()

    found = sorted(result["currency"].unique())
    missing = sorted(set(G7_CURRENCY_FILTERS) - set(found))
    logger.info("Gefundene Waehrungen: %s", found)
    if missing:
        logger.warning("Fehlende Waehrungen: %s", missing)

    return result


def extract_cot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrahiert und berechnet alle CoT-Features."""
    missing_cols = [c for c in COLUMN_MAP if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Fehlende Spalten: {missing_cols}. "
            f"Vorhandene Spalten: {list(df.columns)}"
        )

    result = df[["currency"] + list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP).copy()
    result["date"] = pd.to_datetime(result["date"])

    # Abgeleitete Metriken
    result["net_position"] = result["non_comm_long"] - result["non_comm_short"]
    result["net_change"] = result["change_long"] - result["change_short"]
    # Long-Ratio: Anteil Longs an (Longs + Shorts), epsilon verhindert Division durch 0
    result["long_ratio"] = result["non_comm_long"] / (
        result["non_comm_long"] + result["non_comm_short"] + 1e-9
    )

    return result.sort_values(["currency", "date"]).reset_index(drop=True)


# ── Perzentil-Berechnung ───────────────────────────────────────────────────

def calculate_net_percentile(df: pd.DataFrame, window: int = PERCENTILE_WINDOW) -> pd.DataFrame:
    """
    Berechnet rollendes 3-Jahres-Perzentil der Net Position pro Waehrung.

    Perzentil = Anteil historischer Werte im Fenster, die KLEINER als aktueller Wert sind.
    Erste Zeile pro Waehrung (keine Historie) = 50.0 (neutral).
    """
    df = df.copy()

    def _group_percentile(group: pd.DataFrame) -> pd.Series:
        net = group["net_position"].values
        pct = np.full(len(net), 50.0)
        for i in range(1, len(net)):
            start = max(0, i - window)
            history = net[start:i]
            pct[i] = float(np.mean(net[i] > history) * 100)
        return pd.Series(pct, index=group.index)

    df["net_percentile"] = df.groupby("currency", group_keys=False).apply(_group_percentile)
    return df


# ── Speichern ──────────────────────────────────────────────────────────────

def save_to_parquet(df: pd.DataFrame, out_dir: Path = DATA_DIR) -> Path:
    """Speichert verarbeitete CoT-Daten als Parquet-Datei."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime("%Y%m%d")
    path = out_dir / f"cot_g7_{today}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Gespeichert: %d Zeilen → %s", len(df), path)
    return path


# ── Pipeline ───────────────────────────────────────────────────────────────

def run(url: str = CFTC_URL, out_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Vollstaendige Pipeline: Download → Filter → Extraktion → Perzentil → Parquet."""
    raw = download_raw_cot(url)
    g7 = filter_g7_currencies(raw)
    features = extract_cot_features(g7)
    final = calculate_net_percentile(features)
    save_to_parquet(final, out_dir)
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run()
    summary = (
        result.groupby("currency")
        .tail(1)[["currency", "date", "net_position", "net_percentile", "long_ratio"]]
        .sort_values("currency")
    )
    print("\n=== Letzte CoT-Werte pro Waehrung ===")
    print(summary.to_string(index=False))
