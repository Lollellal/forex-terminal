"""
Gruppe A: Trainings-Pipeline.

Laedt Feature-Daten + FX-Targets, bereitet das Dataset vor,
fuehrt ein 80/20 zeitliches Split durch und trainiert alle 5 Gruppe-A-Modelle.

Ablauf:
  1. Features laden  (data/features/)
  2. Targets laden   (data/raw/fx/ oder FRED)
  3. Abgeleitete Features hinzufuegen  (a_rate_event_upcoming etc.)
  4. Zeitlicher Split: Train 0–80%, Test 80–100% (kein zufaelliges Shuffling)
  5. Jedes Modell trainieren + evaluieren
  6. Modelle + Metadaten speichern (models/group_a/)

Walk-Forward-Validation kommt in Schritt 6.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.group_a.models import (
    GROUP_A_MODEL_CLASSES,
    MODELS_DIR,
    BaseForexModel,
    TrainMetrics,
)
from src.models.group_a.target_builder import (
    HORIZON_DAYS,
    NEUTRAL_THRESHOLD,
    FX_DATA_DIR,
    load_or_fetch_targets,
)

logger = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
TEST_FRACTION = 0.20
MIN_TRAIN_SAMPLES = 100    # Modell wird uebersprungen wenn weniger Samples


# ── Feature-Vorbereitung ───────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet abgeleitete Features fuer Sparse-Signale.

    a_rate_expected_change_filled — NaN durch 0 ersetzen
                                    (kein Event anstehend = keine Erwartung)
    a_rate_event_upcoming         — 1 wenn Rate-Event in 14 Tagen ansteht, sonst 0
    """
    df = df.copy()
    df["a_rate_event_upcoming"] = df["a_rate_expected_change"].notna().astype(float)
    df["a_rate_expected_change_filled"] = df["a_rate_expected_change"].fillna(0.0)
    return df


# ── Dataset-Erstellung ─────────────────────────────────────────────────────

def load_features(features_dir: Path = FEATURES_DIR) -> pd.DataFrame:
    """Laedt die neueste Feature-Parquet-Datei."""
    files = sorted(features_dir.glob("features_g7_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Keine Feature-Datei in {features_dir}. "
            "Bitte zuerst feature_engineer.run() ausfuehren."
        )
    path = files[-1]
    logger.info("Features geladen: %s", path)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_dataset(
    features_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fuegt Features und Targets zusammen.

    features_df: aus feature_engineer (date, currency, a_*, b_*, e_*, f_*)
    target_df:   aus target_builder   (date, currency, target_30d, fx_return_30d)
    """
    df = features_df.merge(
        target_df[["date", "currency", "target_30d", "fx_return_30d"]],
        on=["date", "currency"],
        how="inner",
    )
    df = add_derived_features(df)
    df = df.rename(columns={"target_30d": "_target"})
    df = df.sort_values(["date", "currency"]).reset_index(drop=True)
    logger.info(
        "Dataset: %d Zeilen, %d Waehrungen, Zeitraum %s–%s",
        len(df),
        df["currency"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def temporal_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Teilt den DataFrame zeitlich auf.

    Trainingsset: frueheste (1 - test_fraction) × 100 % der Datenpunkte
    Testset:      juengste test_fraction × 100 % der Datenpunkte

    Kein Shuffling — verhindert Look-Ahead-Bias.
    Gibt (train_df, test_df, cutoff_date) zurueck.
    """
    dates = df["date"].sort_values().unique()
    cutoff_idx  = int(len(dates) * (1 - test_fraction))
    cutoff_date = pd.Timestamp(dates[cutoff_idx])

    train_df = df[df["date"] < cutoff_date].copy()
    test_df  = df[df["date"] >= cutoff_date].copy()

    logger.info(
        "Split: Train %d Zeilen (bis %s), Test %d Zeilen (ab %s)",
        len(train_df), cutoff_date.date(),
        len(test_df),  cutoff_date.date(),
    )
    return train_df, test_df, cutoff_date


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model: BaseForexModel, test_df: pd.DataFrame) -> dict:
    """
    Evaluiert ein trainiertes Modell auf dem Testset.

    Metriken:
      test_accuracy        — Anteil korrekt vorhergesagter Labels (alle Klassen)
      directional_accuracy — Genauigkeit nur auf nicht-neutralen Samples
      n_test               — Anzahl gueltiger Test-Samples
    """
    cols = model.feature_cols + ["_target"]
    valid = test_df[cols].dropna()
    if valid.empty:
        logger.warning("%s: Keine gueltigen Test-Samples", model.model_id)
        return {"test_accuracy": float("nan"), "directional_accuracy": float("nan"), "n_test": 0}

    X = valid[model.feature_cols].to_numpy(dtype=float)
    y_true = valid["_target"].to_numpy(dtype=int)
    y_pred = model._pipeline.predict(X)  # noqa: SLF001

    test_acc = float(np.mean(y_pred == y_true))

    # Directional accuracy: nur auf echten Bewegungen (nicht NEUTRAL)
    non_neutral = (y_true != 0)
    if non_neutral.sum() > 0:
        dir_acc = float(np.mean(y_pred[non_neutral] == y_true[non_neutral]))
    else:
        dir_acc = float("nan")

    logger.info(
        "%s Test — Acc: %.3f | Dir-Acc: %.3f | N: %d",
        model.model_id, test_acc, dir_acc if not np.isnan(dir_acc) else -1, len(y_true),
    )
    return {"test_accuracy": test_acc, "directional_accuracy": dir_acc, "n_test": len(y_true)}


# ── Trainings-Pipeline ─────────────────────────────────────────────────────

def train_group_a(
    dataset: pd.DataFrame,
    model_dir: Path = MODELS_DIR,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, TrainMetrics]:
    """
    Trainiert alle 5 Gruppe-A-Modelle auf dem uebergebenen Dataset.

    dataset:     kombinierter DataFrame aus build_dataset()
                 muss Spalten _target + alle feature_cols enthalten
    model_dir:   Zielverzeichnis fuer .joblib-Dateien + metadata.json
    test_fraction: Anteil Testset (Standard 20 %)

    Gibt Dict {model_id: TrainMetrics} zurueck.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df, cutoff = temporal_split(dataset, test_fraction)

    all_metrics: dict[str, TrainMetrics] = {}

    for ModelClass in GROUP_A_MODEL_CLASSES:
        model = ModelClass()
        logger.info("─── Trainiere %s ───", model.model_id)

        # Genug Samples?
        available = train_df[model.feature_cols + ["_target"]].dropna()
        if len(available) < MIN_TRAIN_SAMPLES:
            logger.warning(
                "%s: nur %d Samples nach NaN-Drop — uebersprungen",
                model.model_id, len(available),
            )
            continue

        try:
            train_m  = model.fit(train_df)
            test_m   = evaluate_model(model, test_df)
            model.save(model_dir)

            metrics = TrainMetrics(
                model_id             = model.model_id,
                n_train              = train_m["n_train"],
                n_test               = test_m["n_test"],
                train_accuracy       = train_m["train_accuracy"],
                test_accuracy        = test_m["test_accuracy"],
                directional_accuracy = test_m["directional_accuracy"],
                feature_importance   = model.feature_importance,
                cutoff_date          = str(cutoff.date()),
            )
            all_metrics[model.model_id] = metrics

        except (ValueError, RuntimeError) as exc:
            logger.error("Fehler bei %s: %s", model.model_id, exc)

    _save_metadata(all_metrics, cutoff, len(train_df), len(test_df), model_dir)
    return all_metrics


def _save_metadata(
    metrics: dict[str, TrainMetrics],
    cutoff: pd.Timestamp,
    n_train: int,
    n_test:  int,
    model_dir: Path,
) -> None:
    """Schreibt Trainings-Metadaten als JSON in das Modell-Verzeichnis."""
    meta = {
        "group":        "A",
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "cutoff_date":  str(cutoff.date()),
        "n_train_rows": n_train,
        "n_test_rows":  n_test,
        "models": {
            mid: {
                "n_train":              m.n_train,
                "n_test":               m.n_test,
                "train_accuracy":       round(m.train_accuracy, 4),
                "test_accuracy":        round(m.test_accuracy, 4)
                    if not np.isnan(m.test_accuracy) else None,
                "directional_accuracy": round(m.directional_accuracy, 4)
                    if not np.isnan(m.directional_accuracy) else None,
                "top_features":         sorted(
                    m.feature_importance.items(), key=lambda x: -x[1]
                )[:3],
            }
            for mid, m in metrics.items()
        },
    }
    path = model_dir / "metadata.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("Metadaten gespeichert → %s", path)


# ── Einstiegspunkt ─────────────────────────────────────────────────────────

def run(
    features_dir: Path = FEATURES_DIR,
    fx_dir:       Path = FX_DATA_DIR,
    model_dir:    Path = MODELS_DIR,
    test_fraction: float = TEST_FRACTION,
    force_refresh: bool = False,
) -> dict[str, TrainMetrics]:
    """
    Vollstaendige Trainings-Pipeline fuer Gruppe A.

    1. Features laden
    2. FX-Targets laden (oder von FRED herunterladen)
    3. Dataset bauen
    4. Alle 5 Modelle trainieren
    5. Modelle + Metadaten speichern
    """
    logger.info("=== Gruppe A Training startet ===")

    features_df = load_features(features_dir)
    target_df   = load_or_fetch_targets(fx_dir, force_refresh=force_refresh)
    dataset     = build_dataset(features_df, target_df)

    metrics = train_group_a(dataset, model_dir=model_dir, test_fraction=test_fraction)

    logger.info("=== Gruppe A Training abgeschlossen ===")
    for mid, m in metrics.items():
        logger.info(
            "  %-28s  Train-Acc: %.3f  Test-Acc: %s  Dir-Acc: %s",
            mid,
            m.train_accuracy,
            f"{m.test_accuracy:.3f}" if not np.isnan(m.test_accuracy)  else "n/a",
            f"{m.directional_accuracy:.3f}" if not np.isnan(m.directional_accuracy) else "n/a",
        )
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run()
    print("\n=== Gruppe A — Trainings-Ergebnisse ===")
    for mid, m in results.items():
        print(
            f"  {mid:<28}  "
            f"Train: {m.train_accuracy:.1%}  "
            f"Test: {m.test_accuracy:.1%}  "
            f"Dir: {m.directional_accuracy:.1%}"
        )
