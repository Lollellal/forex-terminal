"""
Scanner Backtest OOS — Leakage-freie Version

Unterschied zu backtest_scanner.py:
  - ML-Scores kommen NICHT von live-trainierten Modellen (Data Leakage)
  - ML-Scores kommen aus wf_predictions.parquet (echte OOS, walk-forward)
  - Für Woche W werden nur Predictions verwendet, die von einem Modell
    stammen, das auf Daten bis < W trainiert wurde.

Saisonalität: unverändert (walk-forward, kein Leakage)
FX-Returns:   unverändert

Ausgabe:
  - Konsole: Vergleichstabellen
  - reports/scanner_backtest_oos.csv
  - reports/scanner_backtest_oos_summary.csv
"""

import sys
import logging
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Konstanten (identisch zu backtest_scanner.py) ──────────────────────────
G7 = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"]

TOP_12_MODELS = {
    "d4_wachstum_composite", "g2_equity_trend", "e4_dxy_signal",
    "b2_cpi_trend", "f4_yield_spread", "d3_gdp_differenz",
    "d2_gdp_momentum", "a5_zinsuberraschung", "c1_unemp_absolut",
    "a1_leitzins_absolut", "b4_cpi_differenz", "e3_combined_signal",
}

CURRENCY_WEIGHTS = {
    "a1_leitzins_absolut":    {"EUR":0.984,"GBP":1.028,"JPY":1.042,"CAD":1.037,"CHF":0.993,"AUD":0.910,"NZD":1.007},
    "a2_zinsdifferenz":       {"EUR":1.000,"GBP":1.000,"JPY":0.934,"CAD":1.027,"CHF":0.968,"AUD":0.000,"NZD":1.065},
    "a3_zinserwartung":       {"EUR":1.038,"GBP":0.920,"JPY":1.001,"CAD":1.086,"CHF":0.000,"AUD":0.933,"NZD":1.020},
    "a5_zinsuberraschung":    {"EUR":0.992,"GBP":1.043,"JPY":1.038,"CAD":1.000,"CHF":0.957,"AUD":0.000,"NZD":0.966},
    "b1_cpi_absolut_vs_ziel": {"EUR":1.002,"GBP":0.984,"JPY":1.007,"CAD":0.928,"CHF":0.000,"AUD":1.101,"NZD":0.975},
    "b2_cpi_trend":           {"EUR":0.952,"GBP":0.989,"JPY":0.996,"CAD":0.940,"CHF":1.051,"AUD":1.140,"NZD":0.933},
    "b3_cpi_surprise":        {"EUR":1.019,"GBP":0.896,"JPY":1.008,"CAD":0.000,"CHF":0.975,"AUD":1.107,"NZD":0.000},
    "b4_cpi_differenz":       {"EUR":0.896,"GBP":1.055,"JPY":0.919,"CAD":0.953,"CHF":0.000,"AUD":0.962,"NZD":1.206},
    "c1_unemp_absolut":       {"EUR":1.047,"GBP":0.966,"JPY":0.969,"CAD":1.025,"CHF":1.150,"AUD":0.925,"NZD":0.919},
    "c2_unemp_trend":         {"EUR":1.044,"GBP":0.937,"JPY":0.937,"CAD":1.031,"CHF":1.046,"AUD":0.981,"NZD":1.023},
    "c3_unemp_differenz":     {"EUR":0.972,"GBP":0.917,"JPY":1.088,"CAD":1.000,"CHF":1.084,"AUD":0.949,"NZD":0.993},
    "c4_unemp_arbeitsmarkt":  {"EUR":0.960,"GBP":1.059,"JPY":1.063,"CAD":0.938,"CHF":1.069,"AUD":0.891,"NZD":1.018},
    "d1_gdp_absolut":         {"EUR":0.000,"GBP":1.049,"JPY":1.060,"CAD":1.009,"CHF":0.860,"AUD":1.079,"NZD":0.948},
    "d2_gdp_momentum":        {"EUR":1.040,"GBP":0.978,"JPY":0.940,"CAD":1.018,"CHF":1.057,"AUD":1.064,"NZD":0.903},
    "d3_gdp_differenz":       {"EUR":0.926,"GBP":0.987,"JPY":1.078,"CAD":0.999,"CHF":1.106,"AUD":0.962,"NZD":0.944},
    "d4_wachstum_composite":  {"EUR":1.025,"GBP":1.087,"JPY":1.016,"CAD":1.053,"CHF":0.941,"AUD":0.951,"NZD":0.926},
    "e1_commercial_momentum": {"EUR":0.949,"GBP":0.945,"JPY":0.966,"CAD":0.990,"CHF":0.946,"AUD":1.085,"NZD":1.119},
    "e2_small_spec_kontra":   {"EUR":1.003,"GBP":0.986,"JPY":1.030,"CAD":0.977,"CHF":0.993,"AUD":1.020,"NZD":0.000},
    "e3_combined_signal":     {"EUR":0.936,"GBP":1.136,"JPY":0.981,"CAD":1.007,"CHF":1.003,"AUD":0.953,"NZD":0.980},
    "e4_dxy_signal":          {"EUR":0.992,"GBP":1.056,"JPY":0.973,"CAD":0.946,"CHF":1.106,"AUD":0.909,"NZD":1.018},
    "f1_carry_trade":         {"EUR":0.974,"GBP":1.026,"JPY":0.928,"CAD":1.080,"CHF":0.921,"AUD":0.906,"NZD":1.162},
    "f2_esi":                 {"EUR":1.126,"GBP":1.013,"JPY":0.922,"CAD":0.922,"CHF":1.017,"AUD":1.038,"NZD":0.957},
    "f3_handelsbilanz_proxy": {"EUR":0.978,"GBP":1.030,"JPY":0.000,"CAD":0.954,"CHF":0.949,"AUD":1.047,"NZD":1.027},
    "f4_yield_spread":        {"EUR":0.991,"GBP":0.993,"JPY":1.050,"CAD":1.001,"CHF":0.941,"AUD":1.043,"NZD":0.980},
    "g1_vix_risk_off":        {"EUR":0.991,"GBP":0.931,"JPY":1.035,"CAD":1.148,"CHF":0.961,"AUD":1.001,"NZD":0.935},
    "g2_equity_trend":        {"EUR":1.028,"GBP":0.974,"JPY":1.018,"CAD":0.989,"CHF":0.945,"AUD":1.055,"NZD":0.991},
    "g4_yield_curve":         {"EUR":1.020,"GBP":0.965,"JPY":1.059,"CAD":0.961,"CHF":0.973,"AUD":1.029,"NZD":0.994},
    "h1_oil":                 {"EUR":1.047,"GBP":0.000,"JPY":0.984,"CAD":0.990,"CHF":1.020,"AUD":0.978,"NZD":0.000},
    "h2_copper_metals":       {"EUR":1.067,"GBP":1.047,"JPY":0.959,"CAD":0.988,"CHF":0.965,"AUD":0.979,"NZD":0.993},
}

E3_DIVERGENCE_THRESHOLD = 40.0
H2_COPPER_CURRENCIES    = {"CAD", "AUD", "NZD"}
H2_COPPER_WEIGHT        = 1.2
DIR_SCORE               = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}

PAIRS_G7 = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
    "EURGBP","EURJPY","EURCHF","EURCAD","EURAUD","EURNZD",
    "GBPJPY","GBPCHF","GBPCAD","GBPAUD","GBPNZD",
    "AUDJPY","AUDCAD","AUDNZD","CADJPY","CHFJPY",
    "NZDJPY","NZDCHF","NZDCAD",
]

WF_PRED_FILE   = ROOT / "reports" / "wf_predictions.parquet"
BACKTEST_START = date(2018, 1, 1)


# ── Hilfsfunktionen ────────────────────────────────────────────────────────

def iso_week(d: date) -> int:
    return d.isocalendar()[1]


def all_mondays(start: date, end: date) -> list[date]:
    d = start
    while d.weekday() != 0:
        d += timedelta(days=1)
    result = []
    while d <= end:
        result.append(d)
        d += timedelta(weeks=1)
    return result


def pair_quality(ml_score: float, base_abs: float, quote_abs: float) -> tuple[str, float, float]:
    spread  = abs(ml_score)
    avg_abs = (base_abs + quote_abs) / 2

    if spread < 0.12:
        q = "INVALID"
    elif spread < 0.30:
        q = "WEAK"
    elif spread < 0.50:
        q = "VALID"
    else:
        q = "STRONG"

    _DOWNGRADE = {"STRONG": "VALID", "VALID": "WEAK", "WEAK": "INVALID"}
    if avg_abs < 0.15 and q != "INVALID":
        q = _DOWNGRADE[q]

    return q, spread, avg_abs


# ── Step 1: OOS Predictions laden und indexieren ───────────────────────────

def load_oos_predictions() -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Lädt wf_predictions.parquet und gibt ein Dict {date → DataFrame} zurück.
    Bei Duplikaten (Fenster-Grenzen) wird das höhere window behalten.
    """
    print("Lade OOS Predictions...", flush=True)
    df = pd.read_parquet(WF_PRED_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # Duplikate an Fenstergrenzen: höchstes window gewinnt
    df = (
        df.sort_values("window")
          .drop_duplicates(subset=["date", "model_id", "currency"], keep="last")
    )

    # Index: date → DataFrame slice (als dict für O(1) Lookup)
    pred_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for dt, grp in df.groupby("date"):
        pred_by_date[dt] = grp.reset_index(drop=True)

    dates = sorted(pred_by_date.keys())
    print(f"  {len(df):,} Predictions  |  {len(dates)} Wochen  |  "
          f"{dates[0].date()} bis {dates[-1].date()}", flush=True)
    return pred_by_date


def get_nearest_pred_date(
    pred_by_date: dict,
    monday: date,
    max_lookback: int = 7,
) -> pd.Timestamp | None:
    """
    Gibt das nächste verfügbare Prediction-Datum <= monday zurück.
    max_lookback: maximale Rücksuchdistanz in Tagen.
    """
    ts = pd.Timestamp(monday)
    for delta in range(max_lookback + 1):
        candidate = ts - pd.Timedelta(days=delta)
        if candidate in pred_by_date:
            return candidate
    return None


# ── Step 2: ML-Score aus OOS Predictions ──────────────────────────────────

def compute_ml_scores_oos(
    preds: pd.DataFrame,
    e3_divergence: dict[str, float],
    regime: str = "TREND",
) -> dict[str, float]:
    """
    Aggregiert OOS Model-Predictions zu einem gewichteten Currency-Score.
    Regime-Gewichtung: get_regime_weight skaliert pro Modell basierend auf
    historischer Accuracy in diesem Regime (kein Leakage — VIX/YC bekannt).
    """
    from src.backtesting.regime_classifier import get_regime_weight

    scores: dict[str, float] = {}

    # h2_copper Lookup (separat, außerhalb TOP_12)
    h2_rows = preds[preds["model_id"] == "h2_copper_metals"]
    h2_preds: dict[str, str] = {}
    for _, row in h2_rows.iterrows():
        if row["currency"] in H2_COPPER_CURRENCIES:
            h2_preds[row["currency"]] = row["direction"]

    for ccy in G7:
        ccy_preds = preds[preds["currency"] == ccy]
        if ccy_preds.empty:
            continue

        weighted_sum = 0.0
        total_weight = 0.0

        for _, row in ccy_preds.iterrows():
            mid = row["model_id"]
            if mid not in TOP_12_MODELS:
                continue

            if mid == "e3_combined_signal":
                if abs(e3_divergence.get(ccy, 0.0)) < E3_DIVERGENCE_THRESHOLD:
                    continue

            w = CURRENCY_WEIGHTS.get(mid, {}).get(ccy, 1.0)
            if w == 0:
                continue

            # Regime-Gewichtung: skaliert Modell-Beitrag je nach Marktregime
            w *= get_regime_weight(mid, regime)

            weighted_sum += DIR_SCORE.get(row["direction"], 0) * w
            total_weight += w

        # h2_copper Bonus für CAD/AUD/NZD
        if ccy in H2_COPPER_CURRENCIES and ccy in h2_preds:
            weighted_sum += DIR_SCORE.get(h2_preds[ccy], 0) * H2_COPPER_WEIGHT
            total_weight += H2_COPPER_WEIGHT

        if total_weight > 0:
            scores[ccy] = weighted_sum / total_weight

    return scores


def compute_pair_confidence_oos(
    preds: pd.DataFrame,
    base_ccy: str,
    quote_ccy: str,
) -> float:
    """
    Durchschnittliche Modell-Confidence (max(bull,bear)) für ein Pair.
    Nur TOP_12_MODELS.
    """
    top_preds = preds[preds["model_id"].isin(TOP_12_MODELS)]
    pair_preds = top_preds[top_preds["currency"].isin([base_ccy, quote_ccy])]
    if pair_preds.empty:
        return 0.5
    conf = pair_preds[["bull_proba", "bear_proba"]].max(axis=1)
    return float(conf.mean())


# ── Step 3: Features laden (nur für E3 Divergenz) ─────────────────────────

def load_e3_divergence_history() -> pd.DataFrame:
    """
    Lädt historische e_cot_divergence-Werte aus dem Feature-File.
    Kein Leakage — ist Fundamentaldaten, kein Modell-Output.
    """
    feat_dir = ROOT / "data" / "features"
    feat_files = sorted(feat_dir.glob("*.parquet"))
    if not feat_files:
        print("  WARNUNG: Keine Feature-Dateien → E3 Gating deaktiviert", flush=True)
        return pd.DataFrame()

    df = pd.read_parquet(feat_files[-1])
    df["date"] = pd.to_datetime(df["date"])

    if "e_cot_divergence" not in df.columns:
        print("  WARNUNG: e_cot_divergence nicht in Features → E3 Gating deaktiviert", flush=True)
        return pd.DataFrame()

    return df[["date", "currency", "e_cot_divergence"]].dropna()


def get_e3_divergence(
    e3_hist: pd.DataFrame,
    monday: date,
) -> dict[str, float]:
    if e3_hist.empty:
        return {}
    slice_ = e3_hist[e3_hist["date"] <= pd.Timestamp(monday)]
    if slice_.empty:
        return {}
    latest = (
        slice_.sort_values("date")
              .groupby("currency")["e_cot_divergence"]
              .last()
    )
    return latest.to_dict()


# ── Step 3b: Regime History laden ─────────────────────────────────────────

def load_regime_history() -> dict[pd.Timestamp, str]:
    """
    Lädt VIX + Yield Curve aus Feature-Daten und klassifiziert Regime pro Tag.
    Kein Leakage — VIX und Yield Curve sind am Montag bekannt.
    """
    from src.backtesting.regime_classifier import classify_regime

    feat_dir = ROOT / "data" / "features"
    feat_files = sorted(feat_dir.glob("*.parquet"))
    if not feat_files:
        print("  WARNUNG: Keine Feature-Dateien → Regime-Filter deaktiviert", flush=True)
        return {}

    df = pd.read_parquet(feat_files[-1])
    df["date"] = pd.to_datetime(df["date"])

    if "g_vix_level" not in df.columns or "g_yield_curve_us" not in df.columns:
        print("  WARNUNG: VIX/YC fehlen → Regime-Filter deaktiviert", flush=True)
        return {}

    # Eine Zeile pro Datum reicht (Regime ist marktbreit, nicht währungsspezifisch)
    daily = df.drop_duplicates("date").sort_values("date")
    regime_by_date: dict[pd.Timestamp, str] = {}
    for _, row in daily.iterrows():
        r = classify_regime(
            float(row["g_vix_level"])      if not pd.isna(row["g_vix_level"])      else float("nan"),
            float(row["g_yield_curve_us"]) if not pd.isna(row["g_yield_curve_us"]) else float("nan"),
        )
        regime_by_date[row["date"]] = r

    print(f"Regime History geladen: {len(regime_by_date)} Tage", flush=True)
    return regime_by_date


def get_regime_for_week(regime_history: dict, monday: date) -> str:
    """Gibt das Regime für einen Montag zurück — nimmt den letzten bekannten Wert."""
    ts = pd.Timestamp(monday)
    for delta in range(8):
        candidate = ts - pd.Timedelta(days=delta)
        if candidate in regime_history:
            return regime_history[candidate]
    return "TREND"


# Seasonal Edge Demotion nach Regime
# TREND/RISK_ON: unveraendert | RANGE: -1 Stufe | RISK_OFF: -2 Stufen
_EDGE_ORDER  = ["No Edge", "Weak Edge", "Edge", "Strong Edge"]
_REGIME_DEMOTION = {"TREND": 0, "RISK_ON": 0, "RANGE": 1, "RISK_OFF": 2}


def apply_regime_to_edge(edge: str, regime: str) -> str:
    """Demotiert Edge-Klasse basierend auf Regime — ohne Hardcoding auf Jahre."""
    demotion = _REGIME_DEMOTION.get(regime, 0)
    if demotion == 0:
        return edge
    idx = _EDGE_ORDER.index(edge) if edge in _EDGE_ORDER else 0
    return _EDGE_ORDER[max(0, idx - demotion)]


# ── Step 4: Seasonal + Preisdaten ─────────────────────────────────────────

def precompute_seasonal(mondays: list[date]) -> dict:
    """
    Speichert rohe Walk-Forward-Records pro (pair, month, day).
    Edge-Berechnung erfolgt erst beim Lookup mit Date-Cutoff (kein Leakage).
    """
    from src.seasonality.fetcher  import load_prices
    from src.seasonality.backtest import run_walk_forward

    unique_dates = sorted({(m.month, m.day) for m in mondays})
    total = len(PAIRS_G7) * len(unique_dates)
    done  = 0

    print(f"Precompute Seasonal: {len(PAIRS_G7)} Pairs × {len(unique_dates)} Wochen = {total}...", flush=True)

    cache: dict = {}
    for pair in PAIRS_G7:
        prices = load_prices(pair)
        if prices is None or len(prices) < 1000:
            continue
        cache[pair] = {}
        for (mo, dy) in unique_dates:
            try:
                records = run_walk_forward(prices, ref_month=mo, ref_day=dy)
                cache[pair][(mo, dy)] = records or []
            except Exception:
                cache[pair][(mo, dy)] = []
            done += 1
            if done % 100 == 0:
                print(f"  {done/total*100:.0f}% ({done}/{total})", flush=True)

    print("  Seasonal Precompute fertig.", flush=True)
    return cache


def get_seasonal_edge(
    cache: dict,
    pair: str,
    monday: date,
) -> dict:
    """
    Berechnet Seasonal Edge mit striktem Date-Cutoff:
    Für Woche W werden nur Jahre < W.year verwendet.
    5y-Fenster: Jahre in [W.year-5, W.year).
    """
    from src.seasonality.backtest import compute_edge_score

    WINDOWS = ["5d", "10d", "20d"]
    ORDER   = ["No Edge", "Weak Edge", "Edge", "Strong Edge"]
    EMPTY   = {"edge_overall": "No Edge", "dir_overall": None,
               "edge_5y": "No Edge",      "dir_5y": None}

    records = cache.get(pair, {}).get((monday.month, monday.day))
    if not records:
        return EMPTY.copy()

    cutoff_year = monday.year
    recs_overall = [r for r in records if r.get("year", 0) < cutoff_year]
    recs_5y      = [r for r in recs_overall if r.get("year", 0) >= cutoff_year - 5]

    def _best_edge(recs: list) -> tuple[str, str | None]:
        if not recs:
            return "No Edge", None
        edges  = {w: compute_edge_score(recs, w) for w in WINDOWS}
        best_w = max(WINDOWS, key=lambda w: ORDER.index(edges[w]["edge_class"]))
        return edges[best_w]["edge_class"], edges[best_w].get("direction")

    edge_o, dir_o = _best_edge(recs_overall)
    edge_5, dir_5 = _best_edge(recs_5y) if len(recs_5y) >= 3 else ("No Edge", None)

    return {
        "edge_overall": edge_o, "dir_overall": dir_o,
        "edge_5y":      edge_5, "dir_5y":      dir_5,
    }


def load_pair_prices() -> dict[str, pd.Series]:
    from src.seasonality.fetcher import load_prices
    prices = {}
    for pair in PAIRS_G7:
        p = load_prices(pair)
        if p is None or p.empty:
            continue
        if isinstance(p, pd.DataFrame):
            prices[pair] = p["close"] if "close" in p.columns else p.iloc[:, 0]
        else:
            prices[pair] = p
    print(f"Preisdaten geladen: {len(prices)} Pairs", flush=True)
    return prices


def get_5day_return(prices: pd.Series, monday: date) -> float | None:
    friday = monday + timedelta(days=4)
    idx    = prices.index
    mon_dates = idx[idx >= pd.Timestamp(monday)]
    fri_dates = idx[idx >= pd.Timestamp(friday)]
    if len(mon_dates) == 0 or len(fri_dates) == 0:
        return None
    p_mon = prices.loc[mon_dates[0]]
    p_fri = prices.loc[fri_dates[0]]
    if p_mon == 0:
        return None
    return float(p_fri / p_mon - 1)


# ── Step 5: Hauptschleife ──────────────────────────────────────────────────

def run_backtest():
    # OOS-Coverage feststellen
    pred_by_date = load_oos_predictions()
    wf_max_date  = max(pred_by_date.keys()).date()
    backtest_end = min(wf_max_date, date.today() - timedelta(days=7))

    mondays = all_mondays(BACKTEST_START, backtest_end)
    print(f"\nBacktest (OOS): {BACKTEST_START} bis {backtest_end} — {len(mondays)} Wochen\n", flush=True)

    # E3 Divergenz laden
    e3_hist = load_e3_divergence_history()
    print(f"E3 Divergenz: {len(e3_hist)} Zeilen geladen", flush=True)

    # Regime History laden
    regime_history = load_regime_history()

    # Seasonal precomputen
    seasonal_cache = precompute_seasonal(mondays)

    # Preisdaten laden
    pair_prices = load_pair_prices()

    records = []
    skipped_no_pred = 0

    print(f"\nIteriere {len(mondays)} Wochen × {len(PAIRS_G7)} Pairs...", flush=True)

    for i, monday in enumerate(mondays):
        if i % 20 == 0:
            print(f"  Woche {i+1}/{len(mondays)}: {monday}", flush=True)

        # OOS Predictions für diese Woche holen
        pred_date = get_nearest_pred_date(pred_by_date, monday)
        if pred_date is None:
            skipped_no_pred += 1
            continue

        preds = pred_by_date[pred_date]

        # E3 Divergenz für diese Woche
        e3_div = get_e3_divergence(e3_hist, monday)

        # Regime für diese Woche (VIX + Yield Curve, kein Leakage)
        regime = get_regime_for_week(regime_history, monday)

        # ML-Scores berechnen (OOS, mit Regime-Gewichtung)
        ml_scores = compute_ml_scores_oos(preds, e3_div, regime)
        if not ml_scores:
            continue

        for pair in PAIRS_G7:
            base_ccy  = pair[:3]
            quote_ccy = pair[3:]

            base_s  = ml_scores.get(base_ccy)
            quote_s = ml_scores.get(quote_ccy)

            if base_s is None and quote_s is None:
                continue

            ml_score = (base_s or 0.0) - (quote_s or 0.0)
            ml_dir   = "Long" if ml_score >= 0 else "Short"

            q, spread, avg_abs = pair_quality(
                ml_score, abs(base_s or 0.0), abs(quote_s or 0.0)
            )

            if q == "INVALID":
                continue

            # Seasonal (Date-Cutoff: nur Jahre < monday.year)
            sea = get_seasonal_edge(seasonal_cache, pair, monday)

            # Regime-Skalierung: RANGE -1 Stufe, RISK_OFF -2 Stufen
            edge_overall = apply_regime_to_edge(sea["edge_overall"], regime)
            edge_5y      = apply_regime_to_edge(sea["edge_5y"],      regime)
            dir_overall  = sea["dir_overall"]
            dir_5y       = sea["dir_5y"]

            # 5-Tage-Return
            if pair not in pair_prices:
                continue
            ret = get_5day_return(pair_prices[pair], monday)
            if ret is None:
                continue

            directed_ret = ret if ml_dir == "Long" else -ret

            # Pair-Confidence aus Modell-Probas
            pair_conf = compute_pair_confidence_oos(preds, base_ccy, quote_ccy)

            records.append({
                "monday":       monday,
                "pair":         pair,
                "ml_dir":       ml_dir,
                "quality":      q,
                "spread":       round(spread, 3),
                "confidence":   round(pair_conf, 3),
                "regime":       regime,
                "edge_overall": edge_overall,
                "edge_5y":      edge_5y,
                "dir_overall":  dir_overall,
                "dir_5y":       dir_5y,
                "ret_5d":       round(ret * 100, 4),
                "ret_dir":      round(directed_ret * 100, 4),
                "hit":          int(directed_ret > 0),
            })

    if skipped_no_pred > 0:
        print(f"  Übersprungen (kein OOS-Datum): {skipped_no_pred} Wochen", flush=True)

    if not records:
        print("Keine Ergebnisse.")
        return

    df = pd.DataFrame(records)

    # Abgeleitete Spalten vor dem CSV-Save berechnen
    df["edges_agree"]    = df["edge_overall"] == df["edge_5y"]
    df["ml_aligned_5y"]  = df["ml_dir"] == df["dir_5y"]   # ML vs. 5y Seasonal (korrekt)
    df["dirs_agree"]     = df["ml_aligned_5y"]             # Alias für Abwärtskompatibilität

    # CSV sofort sichern (inkl. aller abgeleiteten Spalten)
    out_path = ROOT / "reports" / "scanner_backtest_oos.csv"
    df.to_csv(out_path, index=False)
    print(f"\nRohdaten gespeichert: {out_path}  ({len(df)} Zeilen)", flush=True)

    # ── Ausgabe ───────────────────────────────────────────────────────────
    EDGE_ORD = {"Strong Edge": 3, "Edge": 2, "Weak Edge": 1, "No Edge": 0}
    QUAL_ORD = {"STRONG": 3, "VALID": 2, "WEAK": 1}

    def _stats(grp) -> dict:
        n      = len(grp)
        hit    = grp["hit"].mean() * 100
        avg_r  = grp["ret_dir"].mean()
        std_r  = grp["ret_dir"].std()
        sharpe = (avg_r / std_r * (52**0.5)) if std_r > 0 and n > 2 else 0
        return {"N": n, "Hit%": round(hit, 1), "AvgRet%": round(avg_r, 3), "Sharpe": round(sharpe, 2)}

    print(f"\n{'='*78}")
    print(f"  SCANNER BACKTEST (OOS — leakage-frei)  {BACKTEST_START} – {backtest_end}")
    print(f"  {len(df)} Beobachtungen · {df['monday'].nunique()} Wochen · {df['pair'].nunique()} Pairs")
    print(f"  WF-Predictions: {pred_by_date and min(pred_by_date).date()} – {wf_max_date}")
    print(f"{'='*78}\n")

    # ── 1. ML-Qualität allein ──────────────────────────────────────────────
    print(f"── 1. ML-Qualität (OOS) ───────────────────────────────────────────────")
    print(f"  {'Qualität':8} {'N':>5} {'Hit%':>6} {'AvgRet%':>8} {'Sharpe':>7}")
    print(f"  {'─'*40}")
    for q in ["STRONG", "VALID", "WEAK"]:
        grp = df[df["quality"] == q]
        if grp.empty:
            continue
        s = _stats(grp)
        flag = " *" if s["Hit%"] >= 55 else ""
        print(f"  {q:8} {s['N']:>5} {s['Hit%']:>6.1f} {s['AvgRet%']:>8.3f} {s['Sharpe']:>7.2f}{flag}")

    # ── 2. Overall Edge ────────────────────────────────────────────────────
    print(f"\n── 2. Qualität × Overall Edge ─────────────────────────────────────────")
    print(f"  {'Qualität':8} {'Edge Overall':14} {'N':>5} {'Hit%':>6} {'AvgRet%':>8} {'Sharpe':>7}")
    print(f"  {'─'*52}")
    rows_o = []
    for (q, e), grp in df.groupby(["quality", "edge_overall"]):
        s = _stats(grp)
        rows_o.append({"q": q, "e": e, **s})
    rows_o.sort(key=lambda r: (QUAL_ORD.get(r["q"], 0), EDGE_ORD.get(r["e"], 0)), reverse=True)
    for r in rows_o:
        flag = " *" if r["Hit%"] >= 55 else ""
        print(f"  {r['q']:8} {r['e']:14} {r['N']:>5} {r['Hit%']:>6.1f} {r['AvgRet%']:>8.3f} {r['Sharpe']:>7.2f}{flag}")

    # ── 3. 5y Edge ────────────────────────────────────────────────────────
    print(f"\n── 3. Qualität × 5y Edge ──────────────────────────────────────────────")
    print(f"  {'Qualität':8} {'Edge 5y':14} {'N':>5} {'Hit%':>6} {'AvgRet%':>8} {'Sharpe':>7}")
    print(f"  {'─'*52}")
    rows_5 = []
    for (q, e), grp in df.groupby(["quality", "edge_5y"]):
        s = _stats(grp)
        rows_5.append({"q": q, "e": e, **s})
    rows_5.sort(key=lambda r: (QUAL_ORD.get(r["q"], 0), EDGE_ORD.get(r["e"], 0)), reverse=True)
    for r in rows_5:
        flag = " *" if r["Hit%"] >= 55 else ""
        print(f"  {r['q']:8} {r['e']:14} {r['N']:>5} {r['Hit%']:>6.1f} {r['AvgRet%']:>8.3f} {r['Sharpe']:>7.2f}{flag}")

    # ── 4. ML vs. 5y Seasonal Ausrichtung ────────────────────────────────
    print(f"\n── 4. ML vs. 5y Seasonal Ausrichtung ──────────────────────────────────")
    for label, mask in [
        ("Edge gleich (Ov=5y)",   df["edges_agree"]),
        ("Edge verschieden",      ~df["edges_agree"]),
        ("ML ALIGNED Seasonal",   df["ml_aligned_5y"] & df["dir_5y"].notna()),
        ("ML CONTRARY Seasonal",  ~df["ml_aligned_5y"] & df["dir_5y"].notna()),
    ]:
        grp = df[mask]
        if grp.empty:
            continue
        s = _stats(grp)
        flag = " *" if s["Hit%"] >= 55 else ""
        print(f"  {label:22} N={s['N']:>5}  Hit%={s['Hit%']:.1f}  AvgRet={s['AvgRet%']:.3f}%  Sharpe={s['Sharpe']:.2f}{flag}")

    # ── 5. Spread-Kalibrierung ─────────────────────────────────────────────
    print(f"\n── 5. Spread-Kalibrierung (OOS Thresholds) ────────────────────────────")
    print(f"  {'Spread':10} {'N':>5} {'Hit%':>6} {'AvgRet%':>8} {'Sharpe':>7}")
    for lo, hi in [(0.12, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 99)]:
        grp = df[(df["spread"] >= lo) & (df["spread"] < hi)]
        if grp.empty:
            continue
        s = _stats(grp)
        label = f"{lo:.2f}–{hi:.2f}" if hi < 99 else f"≥{lo:.2f}"
        flag = " *" if s["Hit%"] >= 55 else ""
        print(f"  {label:10} {s['N']:>5} {s['Hit%']:>6.1f} {s['AvgRet%']:>8.3f} {s['Sharpe']:>7.2f}{flag}")

    # ── 6. Confidence-Kalibrierung (NEU — nicht in leaky Scanner) ─────────
    print(f"\n── 6. Confidence-Kalibrierung (OOS, Modell-Probas) ────────────────────")
    print(f"  {'Confidence':12} {'N':>5} {'Hit%':>6} {'AvgRet%':>8} {'Sharpe':>7} {'%Signal':>8}")
    total_n = len(df)
    for lo in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        grp = df[df["confidence"] >= lo]
        if len(grp) < 50:
            break
        s = _stats(grp)
        pct = len(grp) / total_n * 100
        flag = " *" if s["Hit%"] >= 55 else ""
        print(f"  >= {lo:.0%}      {s['N']:>5} {s['Hit%']:>6.1f} {s['AvgRet%']:>8.3f} {s['Sharpe']:>7.2f} {pct:>7.1f}%{flag}")

    # ── 7. VALID × Strong Edge × Confidence kombiniert ────────────────────
    print(f"\n── 7. VALID + Strong Edge × Confidence (bestes Segment) ──────────────")
    print(f"  {'Confidence':12} {'N':>5} {'Hit%':>6} {'AvgRet%':>8} {'Sharpe':>7}")
    valid_strong = df[
        (df["quality"] == "VALID") &
        (df["edge_5y"].isin(["Strong Edge", "Edge"]))
    ]
    for lo in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        grp = valid_strong[valid_strong["confidence"] >= lo]
        if len(grp) < 20:
            break
        s = _stats(grp)
        flag = " *" if s["Hit%"] >= 55 else ""
        print(f"  >= {lo:.0%}      {s['N']:>5} {s['Hit%']:>6.1f} {s['AvgRet%']:>8.3f} {s['Sharpe']:>7.2f}{flag}")

    print(f"\n  * = Hit% >= 55%  |  AvgRet% = richtungsbereinigter Wochenreturn")
    print(f"  OOS: Predictions von walk-forward trainierten Modellen (kein Leakage)\n")

    # ── 8. Direktionale Sweetspots (Quality × 5y Edge × ML Dir × Alignment) ─
    print(f"\n── 8. Direktionale Sweetspots (Quality × 5y Edge × ML Dir × Alignment) ──")
    print(f"  {'Qualität':6} {'5y Edge':14} {'ML Dir':6} {'Align':8} {'N':>5} {'Hit%':>6} {'Sharpe':>7}")
    print(f"  {'─'*62}")
    dir_rows = []
    for (q, e5, ml_d, aligned), g in df.groupby(["quality", "edge_5y", "ml_dir", "ml_aligned_5y"]):
        if len(g) < 30:
            continue
        s = _stats(g)
        dir_rows.append({
            "quality": q, "edge_5y": e5, "ml_dir": ml_d,
            "alignment": "ALIGNED" if aligned else "CONTRARY",
            **s
        })
    dir_rows.sort(key=lambda r: -r["Sharpe"])
    for r in dir_rows[:20]:
        flag = " *" if r["Hit%"] >= 57 else ""
        print(f"  {r['quality']:6} {r['edge_5y']:14} {r['ml_dir']:6} {r['alignment']:8} "
              f"{r['N']:>5} {r['Hit%']:>6.1f} {r['Sharpe']:>7.2f}{flag}")

    # ── CSVs speichern ─────────────────────────────────────────────────────
    summary_rows = []
    for r in rows_o:
        summary_rows.append({"dim": "overall", **r})
    for r in rows_5:
        summary_rows.append({"dim": "5y", **r})
    result_df = pd.DataFrame(summary_rows)
    result_df.to_csv(ROOT / "reports" / "scanner_backtest_oos_summary.csv", index=False)

    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(ROOT / "reports" / "scanner_backtest_oos_directional.csv", index=False)

    df.to_csv(out_path, index=False)
    print(f"\n  Gespeichert: {out_path}")
    print(f"  Gespeichert: reports/scanner_backtest_oos_summary.csv")
    print(f"  Gespeichert: reports/scanner_backtest_oos_directional.csv\n")


if __name__ == "__main__":
    run_backtest()
