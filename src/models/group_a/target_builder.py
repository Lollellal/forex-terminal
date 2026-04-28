"""
FX-Target-Builder fuer Gruppe A ML-Modelle.

Laedt taeglliche Wechselkurse von FRED, berechnet 30-Tage-Vorwaertsrenditen
und erzeugt BULLISH (+1) / NEUTRAL (0) / BEARISH (-1) Labels.

Jede Waehrung wird relativ zu USD gemessen.
USD selbst erhaelt das negative Mittel aller anderen Renditen (Basket-Proxy).

Ohne echte FX-Daten (Tests) kann build_targets() direkt mit einer
synthetischen FX-DataFrame aufgerufen werden.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Konstanten ─────────────────────────────────────────────────────────────

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FX_DATA_DIR   = Path("data/raw/fx")

# FRED-Serien und Quotierungsrichtung:
#   +1 = Direktnotiz (USD per Waehrung, hoeher = staerkere Fremdwaehrung)
#   -1 = Indirektnotiz (Fremdwaehrung per USD, hoeher = schwaechere Fremdwaehrung)
FRED_FX_SERIES: dict[str, tuple[str, int]] = {
    "EUR": ("DEXUSEU", +1),
    "GBP": ("DEXUSUK", +1),
    "JPY": ("DEXJPUS", -1),
    "CAD": ("DEXCAUS", -1),
    "CHF": ("DEXSZUS", -1),
    "AUD": ("DEXUSAL", +1),
}

# 30-Tage Forward-Rendite als Zielvariable
HORIZON_DAYS = 30

# Schwellenwert fuer Bedeutsamkeit:
# |rendite| < THRESHOLD → NEUTRAL (0), darueber → BULLISH (+1) oder BEARISH (-1)
NEUTRAL_THRESHOLD = 0.005   # 0.5 % in 30 Tagen


# ── FRED FX Download ───────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.getenv("FRED_API_KEY", "")
    if not key or key == "your_fred_api_key_here":
        raise EnvironmentError(
            "FRED_API_KEY fehlt. In .env eintragen: FRED_API_KEY=<key>"
        )
    return key


def _fetch_fred_series(series_id: str, api_key: str, start: str) -> pd.Series:
    """Ruft eine einzelne FRED-Zeitreihe ab und gibt eine Date-indexierte Series zurueck."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    dates  = pd.to_datetime([o["date"] for o in obs])
    values = [np.nan if o["value"] == "." else float(o["value"]) for o in obs]
    return pd.Series(values, index=dates, name=series_id, dtype="float64")


def fetch_fx_rates(
    start: str = "2000-01-01",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Laedt taeglliche FX-Kurse von FRED fuer alle G7-Waehrungen (ausser USD).

    Rueckgabe: Wide DataFrame (DatetimeIndex, Spalten = Waehrungscodes).
    Alle Serien werden auf USD-Staerke normiert:
      positiver Wert = Waehrung staerker als USD-Baseline.
    """
    if api_key is None:
        api_key = _get_api_key()

    frames: dict[str, pd.Series] = {}
    for currency, (series_id, direction) in FRED_FX_SERIES.items():
        try:
            raw = _fetch_fred_series(series_id, api_key, start)
            # direction=+1: direkter Return (hoeher = staerkere Waehrung)
            # direction=-1: invertiert (hoeher FRED-Wert = schwaechere Waehrung)
            frames[currency] = raw if direction == +1 else (1.0 / raw)
            logger.info("FX geladen: %s (%d Datenpunkte)", currency, raw.notna().sum())
        except Exception as exc:
            logger.warning("FX-Fehler %s (%s): %s", currency, series_id, exc)

    if not frames:
        raise RuntimeError("Keine FX-Kurse geladen — alle FRED-Abfragen fehlgeschlagen.")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df


def save_fx_to_parquet(fx_wide: pd.DataFrame, out_dir: Path = FX_DATA_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"fx_g7_{today}.parquet"
    fx_wide.reset_index().to_parquet(path, index=False)
    logger.info("FX-Kurse gespeichert → %s", path)
    return path


# ── Target-Konstruktion ────────────────────────────────────────────────────

def build_targets(
    fx_wide: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
    threshold: float = NEUTRAL_THRESHOLD,
) -> pd.DataFrame:
    """
    Berechnet 30-Tage Vorwaertsrenditen und erzeugt Klassifikations-Labels.

    Eingabe:   fx_wide — Wide DataFrame (DatetimeIndex, Spalten = Waehrungen)
               Werte repraesentieren USD-normierte Kurse (hoeher = staerkere Waehrung)

    Ausgabe:   Long DataFrame mit Spalten:
               - date (Timestamp)
               - currency (str)
               - fx_return_30d  (float) — tatsaechliche Forward-Rendite
               - target_30d     (int)   — +1 / 0 / -1

    USD-Return: negatives Mittel der anderen Waehrungs-Returns
    (staerkeres Ausland = schwaecheres USD).
    """
    # 30-Tage Forward-Return: (price_t+h / price_t) - 1
    # Shift rueckwaerts, damit jede Zeile ihren ZUKUENFTIGEN Return kennt
    fx_clean = fx_wide.copy()
    fx_clean = fx_clean.ffill(limit=5)           # FRED hat Luecken (Feiertage)

    returns = fx_clean.shift(-horizon_days) / fx_clean - 1.0

    # USD: negatives Mittel der anderen G7-Returns (DXY-Basket-Proxy)
    if "USD" not in returns.columns:
        available_others = [c for c in returns.columns if c != "USD"]
        if available_others:
            returns["USD"] = -returns[available_others].mean(axis=1)

    # Labels
    labels = np.select(
        [returns > threshold, returns < -threshold],
        [1, -1],
        default=0,
    )
    labels_df = pd.DataFrame(labels, index=returns.index, columns=returns.columns)

    # Long-Format zusammenbauen
    records: list[pd.DataFrame] = []
    for currency in returns.columns:
        tmp = pd.DataFrame({
            "date":        returns.index,
            "currency":    currency,
            "fx_return_30d": returns[currency].values,
            "target_30d":   labels_df[currency].values,
        })
        records.append(tmp)

    result = pd.concat(records, ignore_index=True)
    result = result.dropna(subset=["fx_return_30d"])
    result["target_30d"] = result["target_30d"].astype(int)
    result = result.sort_values(["date", "currency"]).reset_index(drop=True)

    n_total = len(result)
    for label, name in [(1, "BULLISH"), (0, "NEUTRAL"), (-1, "BEARISH")]:
        n = (result["target_30d"] == label).sum()
        logger.info("Klassen-Verteilung %s: %d (%.1f%%)", name, n, 100 * n / n_total)

    return result


def load_or_fetch_targets(
    fx_dir: Path = FX_DATA_DIR,
    start: str = "2000-01-01",
    horizon_days: int = HORIZON_DAYS,
    threshold: float = NEUTRAL_THRESHOLD,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Laedt gecachte FX-Daten oder laedt sie von FRED und erstellt Targets.
    """
    cached = sorted(fx_dir.glob("fx_g7_*.parquet"))
    if cached and not force_refresh:
        path = cached[-1]
        logger.info("Lade gecachte FX-Daten: %s", path)
        fx_df = pd.read_parquet(path)
        fx_df["date"] = pd.to_datetime(fx_df["date"])
        fx_wide = fx_df.set_index("date")
    else:
        fx_wide = fetch_fx_rates(start=start)
        save_fx_to_parquet(fx_wide, fx_dir)

    return build_targets(fx_wide, horizon_days=horizon_days, threshold=threshold)
