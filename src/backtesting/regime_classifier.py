"""
Regime Classifier — bestimmt das aktuelle Marktregime täglich.

4 Regime (Priorität: RISK_OFF > RISK_ON > RANGE > TREND):
    RISK_OFF  VIX > 25                         → Flight-to-Safety
    RISK_ON   VIX < 15 AND Yield Curve > 0.3   → Carry/Growth
    RANGE     Yield Curve < 0 (invertiert)     → Rezessionsangst
    TREND     alles andere                     → Normalzustand

Datenquellen (bereits in Feature-Parquet vorhanden):
    g_vix_level      — CBOE VIX täglich
    g_yield_curve_us — US 10Y minus 2Y

Credit Spreads (noch nicht implementiert) wären Schritt 2.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

Regime = Literal["RISK_OFF", "RISK_ON", "RANGE", "TREND"]

# Schwellenwerte
VIX_RISK_OFF  = 25.0   # VIX > 25 → RISK_OFF
VIX_RISK_ON   = 15.0   # VIX < 15 (+ YC > 0.3) → RISK_ON
YC_RISK_ON    =  0.3   # Yield-Kurve Minimum für RISK_ON
YC_RANGE      =  0.0   # YC < 0 → RANGE (invertierte Kurve)

# Regime-Gewichte pro Modell — datengetrieben aus Post-hoc-Backtest (2026-06-09)
# Methode: weight = dir_acc / avg_dir_acc_per_regime, geclampt auf [0.8, 1.5]
# Basis: 50d-Horizont, reports/regime_posthoc_report.md (35 WF-Fenster 2017–2026)
# Runde 7: TOP_12 komplett neu kalibriert (d2, d4, c2, f4, b4 erstmals kalibriert)
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "RISK_ON": {
        # avg dir-acc = 52.4% — deaktiviert: d4 (33.3% = Zufall!), d2 (38.9%)
        "e3_cot_wochen_flow":    1.25,   # 65.6%
        "b4_cpi_differenz":      1.14,   # 59.5%
        "f2_esi":                1.10,   # 57.5%
        "g2_equity_trend":       1.10,   # 57.5%
        "b2_cpi_trend":          1.05,   # 54.8%
        "c2_unemp_trend":        1.04,   # 54.7%
        "f4_yield_spread":       1.02,   # 53.4%
        "a1_leitzins_absolut":   0.97,   # 51.0%
        "g1_vix_risk_off":       0.97,   # 50.9%
        "c1_unemp_absolut":      0.99,   # 51.9%
        "d2_gdp_momentum":       0.00,   # 38.9% → deaktiviert (<48%)
        "d4_wachstum_composite": 0.00,   # 33.3% → deaktiviert (Zufallsniveau)
        "_others":               1.00,
    },
    "TREND": {
        # avg dir-acc = 54.9% — keine Deaktivierungen
        "d4_wachstum_composite": 1.13,   # 62.2%
        "g2_equity_trend":       1.07,   # 58.7%
        "d2_gdp_momentum":       1.07,   # 58.5%
        "f2_esi":                1.02,   # 56.2%
        "a1_leitzins_absolut":   1.02,   # 55.8%
        "b2_cpi_trend":          1.00,   # 55.0%
        "c1_unemp_absolut":      0.97,   # 53.2%
        "b4_cpi_differenz":      0.96,   # 52.6%
        "g1_vix_risk_off":       0.96,   # 52.6%
        "f4_yield_spread":       0.96,   # 52.9%
        "c2_unemp_trend":        0.95,   # 52.1%
        "e3_cot_wochen_flow":    0.90,   # 49.3%
        "_others":               1.00,
    },
    "RANGE": {
        # avg dir-acc = 55.1% — deaktiviert: f2 (40.0%)
        "f4_yield_spread":       1.15,   # 63.5%
        "g1_vix_risk_off":       1.14,   # 62.7%
        "g2_equity_trend":       1.13,   # 62.2%
        "c1_unemp_absolut":      1.05,   # 57.8%
        "d4_wachstum_composite": 1.05,   # 57.7%
        "b4_cpi_differenz":      1.00,   # 55.1%
        "e3_cot_wochen_flow":    0.96,   # 53.1%
        "b2_cpi_trend":          0.97,   # 53.5%
        "c2_unemp_trend":        0.97,   # 53.4%
        "a1_leitzins_absolut":   0.97,   # 53.7%
        "d2_gdp_momentum":       0.88,   # 48.4%
        "f2_esi":                0.00,   # 40.0% → deaktiviert (<48%)
        "_others":               1.00,
    },
    "RISK_OFF": {
        # avg dir-acc = 56.9% — keine Deaktivierungen
        "b2_cpi_trend":          1.12,   # 64.0%
        "d2_gdp_momentum":       1.11,   # 63.4%
        "d4_wachstum_composite": 1.09,   # 62.0%
        "c1_unemp_absolut":      1.04,   # 59.3%
        "e3_cot_wochen_flow":    1.02,   # 58.3%
        "f2_esi":                1.02,   # 58.2%
        "c2_unemp_trend":        1.00,   # 56.7%
        "a1_leitzins_absolut":   0.97,   # 55.0%
        "b4_cpi_differenz":      0.95,   # 54.3%
        "g1_vix_risk_off":       0.91,   # 51.7%
        "g2_equity_trend":       0.89,   # 50.4%
        "f4_yield_spread":       0.88,   # 50.0%
        "_others":               1.00,
    },
}

# Historische Häufigkeit laut Runde-5-Analyse (2010–2026)
REGIME_BASE_RATES = {
    "RISK_ON":  0.21,
    "TREND":    0.48,
    "RANGE":    0.17,
    "RISK_OFF": 0.14,
}


def classify_regime(vix: float, yield_curve: float) -> Regime:
    """
    Bestimmt das Marktregime anhand von VIX und Yield Curve.

    Priorität: RISK_OFF → RISK_ON → RANGE → TREND
    """
    if pd.isna(vix) or pd.isna(yield_curve):
        return "TREND"  # Fallback bei fehlenden Daten
    if vix > VIX_RISK_OFF:
        return "RISK_OFF"
    if vix < VIX_RISK_ON and yield_curve > YC_RISK_ON:
        return "RISK_ON"
    if yield_curve < YC_RANGE:
        return "RANGE"
    return "TREND"


def get_current_regime(features_df: pd.DataFrame) -> tuple[Regime, float, float]:
    """
    Bestimmt das Regime für den letzten verfügbaren Tag.

    Gibt zurück: (regime, vix, yield_curve)
    """
    # Letzten Tag holen (features_df kann mehrere Währungen haben)
    df = features_df.copy()
    if "currency" in df.columns:
        df = df[df["currency"] == df["currency"].iloc[0]]  # eine Währung reicht
    df = df.sort_values("date")

    # Letzten gültigen VIX/YC-Wert suchen
    vix = float("nan")
    yc  = float("nan")

    for col_vix in ("g_vix_level",):
        if col_vix in df.columns:
            series = df[col_vix].dropna()
            if not series.empty:
                vix = float(series.iloc[-1])

    for col_yc in ("g_yield_curve_us",):
        if col_yc in df.columns:
            series = df[col_yc].dropna()
            if not series.empty:
                yc = float(series.iloc[-1])

    regime = classify_regime(vix, yc)
    logger.info(
        "[Regime] VIX=%.1f  YieldCurve=%.2f  → %s",
        vix if not pd.isna(vix) else -1,
        yc  if not pd.isna(yc)  else -1,
        regime,
    )
    return regime, vix, yc


def get_regime_weight(model_id: str, regime: Regime) -> float:
    """Gibt das Regime-Gewicht für ein Modell zurück."""
    weights = REGIME_WEIGHTS.get(regime, {})
    return weights.get(model_id, weights.get("_others", 1.0))


def classify_series(features_df: pd.DataFrame) -> pd.Series:
    """
    Klassifiziert das Regime für alle Tage in features_df.
    Gibt eine pd.Series mit Regime-Labels zurück (Index = date).
    Nützlich für Post-hoc-Backtest-Analyse.
    """
    df = features_df.copy()
    if "currency" in df.columns:
        df = df.drop_duplicates("date").sort_values("date")

    vix = df.get("g_vix_level",      pd.Series(dtype=float))
    yc  = df.get("g_yield_curve_us", pd.Series(dtype=float))

    regimes = pd.Series(index=df["date"], dtype=str)
    for i, row in df.iterrows():
        v = float(vix.get(i, float("nan"))) if hasattr(vix, "get") else float(vix.iloc[i] if i < len(vix) else float("nan"))
        y = float(yc.get(i,  float("nan"))) if hasattr(yc,  "get") else float(yc.iloc[i]  if i < len(yc)  else float("nan"))
        regimes.iloc[list(df.index).index(i)] = classify_regime(v, y)

    return regimes


def regime_stats(features_df: pd.DataFrame) -> dict[str, float]:
    """
    Berechnet die historische Regime-Verteilung.
    Nützlich zur Validierung der Schwellenwerte.
    """
    df = features_df.copy()
    if "currency" in df.columns:
        df = df.drop_duplicates("date").sort_values("date")

    counts: dict[str, int] = {"RISK_ON": 0, "TREND": 0, "RANGE": 0, "RISK_OFF": 0}
    total = 0
    for _, row in df.iterrows():
        vix = row.get("g_vix_level",      float("nan"))
        yc  = row.get("g_yield_curve_us", float("nan"))
        r   = classify_regime(float(vix) if not pd.isna(vix) else float("nan"),
                              float(yc)  if not pd.isna(yc)  else float("nan"))
        counts[r] += 1
        total += 1

    return {k: round(v / total, 3) if total > 0 else 0.0 for k, v in counts.items()}
