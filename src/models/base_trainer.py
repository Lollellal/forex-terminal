"""
Generische Trainings-Pipeline fuer alle Forex ML Modellgruppen.

Stellt bereit:
  load_features      — neueste Feature-Parquet laden
  build_dataset      — Features + Targets zusammenfuehren
  temporal_split     — zeitlicher Train/Test-Split (kein Look-Ahead)
  evaluate_model     — Metriken auf Testset berechnen
  train_group        — alle Modelle einer Gruppe trainieren + speichern
  _save_metadata     — Metadaten-JSON schreiben

Jede Gruppe importiert diese Funktionen und stellt nur ihre
Modell-Klassen-Liste sowie ggf. eine prepare_fn bereit.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.models.group_a.models import BaseForexModel, TrainMetrics

logger = logging.getLogger(__name__)

FEATURES_DIR  = Path("data/features")
TEST_FRACTION = 0.20
MIN_TRAIN_SAMPLES = 100


# ── Feature-Laden ──────────────────────────────────────────────────────────

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


# ── Dataset-Erstellung ─────────────────────────────────────────────────────

def build_dataset(
    features_df: pd.DataFrame,
    target_df: pd.DataFrame,
    horizon: int = 50,
) -> pd.DataFrame:
    """
    Fuegt Features und Targets zusammen.

    features_df: aus feature_engineer (date, currency, a_*, b_*, e_*, …)
    target_df:   aus target_builder   (date, currency, target_{h}d, fx_return_{h}d)
    horizon:     Vorhersage-Horizont in Tagen (10 / 20 / 30 / 50)
    """
    target_col = f"target_{horizon}d"
    return_col = f"fx_return_{horizon}d"

    df = features_df.merge(
        target_df[["date", "currency", target_col, return_col]],
        on=["date", "currency"],
        how="inner",
    )
    # Intern immer "_target" und "fx_return_30d" (WF-Engine erwartet letzteren Namen)
    df = df.rename(columns={target_col: "_target", return_col: "fx_return_30d"})
    df = df.sort_values(["date", "currency"]).reset_index(drop=True)
    logger.info(
        "Dataset (h=%dd): %d Zeilen, %d Waehrungen, Zeitraum %s–%s",
        horizon, len(df), df["currency"].nunique(),
        df["date"].min().date(), df["date"].max().date(),
    )
    return df


# ── Zeitlicher Split ───────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Teilt den DataFrame zeitlich auf (kein Shuffling → kein Look-Ahead-Bias).

    Rueckgabe: (train_df, test_df, cutoff_date)
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

def evaluate_model(model: "BaseForexModel", test_df: pd.DataFrame) -> dict:
    """
    Evaluiert ein trainiertes Modell auf dem Testset.

    Metriken:
      test_accuracy        — Anteil korrekt vorhergesagter Labels
      directional_accuracy — Genauigkeit nur auf nicht-neutralen Samples
      n_test               — Anzahl gueltiger Test-Samples
    """
    cols = model.feature_cols + ["_target"]
    valid = test_df[cols].dropna()
    if valid.empty:
        logger.warning("%s: Keine gueltigen Test-Samples", model.model_id)
        return {"test_accuracy": float("nan"), "directional_accuracy": float("nan"), "n_test": 0}

    X      = valid[model.feature_cols].to_numpy(dtype=float)
    y_true = valid["_target"].to_numpy(dtype=int)
    y_pred = model._pipeline.predict(X)  # noqa: SLF001

    test_acc = float(np.mean(y_pred == y_true))

    non_neutral = (y_true != 0)
    dir_acc = (
        float(np.mean(y_pred[non_neutral] == y_true[non_neutral]))
        if non_neutral.sum() > 0
        else float("nan")
    )

    logger.info(
        "%s Test — Acc: %.3f | Dir-Acc: %s | N: %d",
        model.model_id, test_acc,
        f"{dir_acc:.3f}" if not np.isnan(dir_acc) else "n/a",
        len(y_true),
    )
    return {"test_accuracy": test_acc, "directional_accuracy": dir_acc, "n_test": len(y_true)}


# ── Metadaten ──────────────────────────────────────────────────────────────

def _save_metadata(
    metrics: dict[str, "TrainMetrics"],
    cutoff: pd.Timestamp,
    n_train: int,
    n_test: int,
    model_dir: Path,
    group_letter: str,
) -> None:
    meta = {
        "group":        group_letter,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "cutoff_date":  str(cutoff.date()),
        "n_train_rows": n_train,
        "n_test_rows":  n_test,
        "note": (
            "train_accuracy/test_accuracy/directional_accuracy sind die ehrliche "
            "Out-of-Sample-Referenz aus dem 80/20-Split (Train < cutoff_date, "
            "Test >= cutoff_date). Das gespeicherte .joblib-Modell wurde NACH dieser "
            "Evaluation zusaetzlich auf 100% der verfuegbaren Daten (n_refit Samples) "
            "refittet und kennt damit auch die juengsten Test-Daten -- die OOS-Metriken "
            "oben beziehen sich also auf eine frühere, kleinere Version des Modells, "
            "nicht auf das tatsaechlich gespeicherte."
        ),
        "models": {
            mid: {
                "n_train":              m.n_train,
                "n_test":               m.n_test,
                "n_refit":              m.n_refit,
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


# ── Haupt-Trainings-Pipeline ───────────────────────────────────────────────

def train_group(
    model_classes: list,
    dataset: pd.DataFrame,
    model_dir: Path,
    group_letter: str,
    test_fraction: float = TEST_FRACTION,
    prepare_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, "TrainMetrics"]:
    """
    Trainiert alle Modelle einer Gruppe auf dem uebergebenen Dataset.

    model_classes  — Liste von BaseForexModel-Unterklassen
    dataset        — kombinierter DataFrame aus build_dataset()
    model_dir      — Zielverzeichnis fuer .joblib-Dateien
    group_letter   — "A", "B", … fuer Metadaten-JSON
    test_fraction  — Anteil Testset (Standard 20 %)
    prepare_fn     — optionale Funktion fuer gruppenspezifische abgeleitete Features
                     (z.B. add_derived_features fuer Gruppe A)
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_fn(dataset) if prepare_fn is not None else dataset
    train_df, test_df, cutoff = temporal_split(df, test_fraction)

    all_metrics: dict[str, TrainMetrics] = {}

    for ModelClass in model_classes:
        model = ModelClass()
        logger.info("─── Trainiere %s ───", model.model_id)

        available = train_df[model.feature_cols + ["_target"]].dropna()
        if len(available) < MIN_TRAIN_SAMPLES:
            logger.warning(
                "%s: nur %d Samples nach NaN-Drop — uebersprungen",
                model.model_id, len(available),
            )
            continue

        try:
            train_m = model.fit(train_df)
            test_m  = evaluate_model(model, test_df)

            # Refit auf 100% der Daten NACH der OOS-Evaluation. train_m/test_m bleiben
            # die ehrliche Out-of-Sample-Referenz aus dem 80/20-Split (Cutoff = cutoff);
            # das gespeicherte Modell selbst kennt zusaetzlich die juengsten
            # test_fraction der Daten, statt bei cutoff eingefroren zu bleiben.
            refit_m = model.fit(df)
            model.save(model_dir)

            all_metrics[model.model_id] = TrainMetrics(
                model_id             = model.model_id,
                n_train              = train_m["n_train"],
                n_test               = test_m["n_test"],
                train_accuracy       = train_m["train_accuracy"],
                test_accuracy        = test_m["test_accuracy"],
                directional_accuracy = test_m["directional_accuracy"],
                feature_importance   = model.feature_importance,
                cutoff_date          = str(cutoff.date()),
                n_refit              = refit_m["n_train"],
            )
        except (ValueError, RuntimeError) as exc:
            logger.error("Fehler bei %s: %s", model.model_id, exc)

    _save_metadata(all_metrics, cutoff, len(train_df), len(test_df), model_dir, group_letter)
    return all_metrics
