"""
Gruppe A: Trainings-Pipeline (Zinspolitik & Zentralbank).

Duenner Wrapper um base_trainer — stellt alle bisherigen Exporte bereit
damit bestehende Tests unveraendert weiterlaufen.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base_trainer import (
    build_dataset as _build_dataset_base,
    evaluate_model,
    load_features,
    temporal_split,
    train_group,
)
from src.models.group_a.models import GROUP_A_MODEL_CLASSES, MODELS_DIR, BaseForexModel, TrainMetrics
from src.models.group_a.target_builder import FX_DATA_DIR, HORIZON_DAYS, NEUTRAL_THRESHOLD, load_or_fetch_targets

logger = logging.getLogger(__name__)

TEST_FRACTION = 0.20
FEATURES_DIR  = Path("data/features")


# ── Gruppenspezifische abgeleitete Features ────────────────────────────────

def build_dataset(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Gruppe-A-spezifisch: fuegt derived Features automatisch hinzu."""
    return add_derived_features(_build_dataset_base(features_df, target_df))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet abgeleitete Features fuer Gruppe A.

    a_rate_event_upcoming         — 1 wenn Rate-Event in 14 Tagen ansteht
    a_rate_expected_change_filled — NaN durch 0 ersetzen (kein Event = keine Erwartung)
    """
    df = df.copy()
    df["a_rate_event_upcoming"]         = df["a_rate_expected_change"].notna().astype(float)
    df["a_rate_expected_change_filled"] = df["a_rate_expected_change"].fillna(0.0)
    return df


# ── Training ───────────────────────────────────────────────────────────────

def train_group_a(
    dataset: pd.DataFrame,
    model_dir: Path = MODELS_DIR,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, TrainMetrics]:
    """Trainiert alle 5 Gruppe-A-Modelle. Wrapper um base_trainer.train_group()."""
    return train_group(
        model_classes = GROUP_A_MODEL_CLASSES,
        dataset       = dataset,
        model_dir     = model_dir,
        group_letter  = "A",
        test_fraction = test_fraction,
        prepare_fn    = add_derived_features,
    )


# ── Vollstaendige Pipeline ─────────────────────────────────────────────────

def run(
    features_dir: Path = FEATURES_DIR,
    fx_dir:       Path = FX_DATA_DIR,
    model_dir:    Path = MODELS_DIR,
    test_fraction: float = TEST_FRACTION,
    force_refresh: bool = False,
) -> dict[str, TrainMetrics]:
    """Vollstaendige Trainings-Pipeline fuer Gruppe A."""
    logger.info("=== Gruppe A Training startet ===")

    features_df = load_features(features_dir)
    target_df   = load_or_fetch_targets(fx_dir, force_refresh=force_refresh)
    dataset     = build_dataset(features_df, target_df)
    metrics     = train_group_a(dataset, model_dir=model_dir, test_fraction=test_fraction)

    logger.info("=== Gruppe A Training abgeschlossen ===")
    for mid, m in metrics.items():
        logger.info(
            "  %-28s  Train-Acc: %.3f  Test-Acc: %s  Dir-Acc: %s",
            mid, m.train_accuracy,
            f"{m.test_accuracy:.3f}" if not np.isnan(m.test_accuracy) else "n/a",
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
