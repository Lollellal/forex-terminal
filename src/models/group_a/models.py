"""
Gruppe A: Zinspolitik & Zentralbank — ML-Modelle (A1–A5).

Jedes Modell liefert pro Waehrung:
  - direction:  "BULLISH" | "NEUTRAL" | "BEARISH"
  - confidence: 0–100 (Wahrscheinlichkeit der Gewinner-Klasse × 100)

Algorithmus: Random Forest mit probability calibration.
Der Classifier wird in einer sklearn Pipeline mit StandardScaler verpackt,
sodass Feature-Skalierung automatisch mitgespeichert wird.

Speicherung: joblib → /models/group_a/<model_id>.joblib
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Konstanten ─────────────────────────────────────────────────────────────

MODELS_DIR = Path("models/group_a")

# Label-Mapping fuer den Classifier
LABEL_TO_DIRECTION: dict[int, str] = {
    1: "BULLISH",
    0: "NEUTRAL",
   -1: "BEARISH",
}

# RandomForest-Hyperparameter — konservativ fuer geringe Datenmenge
RF_PARAMS: dict = {
    "n_estimators":    200,
    "max_depth":         5,
    "min_samples_leaf": 20,
    "random_state":     42,
    "class_weight":  "balanced",   # gleicht Klassen-Ungleichgewicht aus
    "n_jobs":           -1,
}

# ── Datenklassen ───────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Vorhersage eines einzelnen Modells fuer eine Waehrung."""
    currency:   str
    direction:  str   # "BULLISH" | "NEUTRAL" | "BEARISH"
    confidence: float # 0–100
    model_id:   str


@dataclass
class TrainMetrics:
    """Trainings- und Test-Metriken eines Modells."""
    model_id:              str
    n_train:               int
    n_test:                int
    train_accuracy:        float
    test_accuracy:         float
    directional_accuracy:  float   # nur auf nicht-neutralen Samples
    feature_importance:    dict[str, float] = field(default_factory=dict)
    cutoff_date:           str = ""


# ── Basis-Klasse ───────────────────────────────────────────────────────────

class BaseForexModel:
    """
    Basis-Klasse fuer alle Gruppe-A-Modelle.

    Unterklassen definieren:
      - model_id:     eindeutiger Bezeichner (z.B. "a1_leitzins_absolut")
      - feature_cols: Liste der Feature-Spaltennamen aus dem Feature-DataFrame
      - description:  Kurzbeschreibung fuer Dokumentation

    fit() / predict() / save() / load() sind in dieser Klasse implementiert.
    """
    model_id:    str = "base"
    feature_cols: list[str] = []
    description: str = ""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_importance: dict[str, float] = {}
        self._is_fitted: bool = False

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(**RF_PARAMS)),
        ])

    def _clean_data(
        self, df: pd.DataFrame, target_col: str = "_target"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Entfernt NaN-Zeilen und gibt (X, y) zurueck."""
        cols = self.feature_cols + [target_col]
        valid = df[cols].dropna()
        X = valid[self.feature_cols].to_numpy(dtype=float)
        y = valid[target_col].to_numpy(dtype=int)
        return X, y

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, target_col: str = "_target") -> dict:
        """
        Trainiert das Modell.

        Erwartet einen DataFrame mit allen feature_cols und der Zielspalte.
        Gibt ein Metriken-Dict zurueck (wird von trainer.py weiterverwendet).
        Wirft ValueError wenn nicht genug Samples vorhanden.
        """
        X, y = self._clean_data(train_df, target_col)
        if len(X) < 30:
            raise ValueError(
                f"{self.model_id}: Zu wenige Trainings-Samples nach NaN-Drop: {len(X)}"
            )

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)

        clf = self._pipeline.named_steps["clf"]
        self._feature_importance = dict(
            zip(self.feature_cols, clf.feature_importances_)
        )
        self._is_fitted = True

        preds = self._pipeline.predict(X)
        train_acc = float(np.mean(preds == y))
        logger.info(
            "%s trainiert — %d Samples, Train-Acc: %.3f", self.model_id, len(X), train_acc
        )
        return {"n_train": len(X), "train_accuracy": train_acc}

    # ── Inferenz ──────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> list[ModelResult]:
        """
        Gibt fuer jede Zeile im DataFrame eine Vorhersage zurueck.

        Zeilen mit fehlenden Features werden uebersprungen (kein Fehler).
        """
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError(f"{self.model_id}: Modell ist nicht trainiert (fit zuerst)")

        results: list[ModelResult] = []
        for _, row in df.iterrows():
            vals = [row.get(col) for col in self.feature_cols]
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
                continue

            X = np.array(vals, dtype=float).reshape(1, -1)
            proba  = self._pipeline.predict_proba(X)[0]
            classes = self._pipeline.classes_

            best_idx   = int(np.argmax(proba))
            best_class = int(classes[best_idx])
            direction  = LABEL_TO_DIRECTION.get(best_class, "NEUTRAL")
            confidence = float(proba[best_idx] * 100)

            results.append(ModelResult(
                currency   = str(row.get("currency", "UNKNOWN")),
                direction  = direction,
                confidence = confidence,
                model_id   = self.model_id,
            ))
        return results

    def predict_latest(self, df: pd.DataFrame) -> list[ModelResult]:
        """Vorhersage nur fuer den juengsten Datenpunkt jeder Waehrung."""
        latest = df.sort_values("date").groupby("currency").last().reset_index()
        return self.predict(latest)

    # ── Persistenz ────────────────────────────────────────────────────────

    def save(self, model_dir: Path = MODELS_DIR) -> Path:
        """Speichert Pipeline + Metadaten als .joblib-Datei."""
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{self.model_id}.joblib"
        payload = {
            "model_id":           self.model_id,
            "feature_cols":       self.feature_cols,
            "description":        self.description,
            "pipeline":           self._pipeline,
            "feature_importance": self._feature_importance,
        }
        joblib.dump(payload, path)
        logger.info("Modell gespeichert → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseForexModel":
        """Laedt ein gespeichertes Modell von Disk."""
        payload    = joblib.load(path)
        obj        = cls()
        obj._pipeline            = payload["pipeline"]
        obj._feature_importance  = payload.get("feature_importance", {})
        obj._is_fitted           = True
        logger.info("Modell geladen: %s", payload.get("model_id", path.stem))
        return obj

    # ── Hilfsmethoden ─────────────────────────────────────────────────────

    @property
    def feature_importance(self) -> dict[str, float]:
        return self._feature_importance

    def __repr__(self) -> str:
        status = "trained" if self._is_fitted else "untrained"
        return f"{self.__class__.__name__}(id={self.model_id}, {status})"


# ── Gruppe A: Modell-Definitionen ──────────────────────────────────────────

class A1LeitzinsAbsolut(BaseForexModel):
    """
    A1 — Leitzins Absolut.

    Bewertet ob das aktuelle Zinsniveau hoch oder niedrig ist.
    Hoeher als historischer Durchschnitt → eher bullish (Kapitalzufluss).
    Features: Absolutes Niveau, 3-Jahres-Durchschnitt, Abweichung.
    """
    model_id     = "a1_leitzins_absolut"
    description  = "A1: Leitzins-Niveau vs. historischem Durchschnitt"
    feature_cols = [
        "a_interest_rate",
        "a_rate_hist_avg_3y",
        "a_rate_dev_from_avg",
    ]


class A2Zinsdifferenz(BaseForexModel):
    """
    A2 — Zinsdifferenz.

    Misst den Leitzins-Spread dieser Waehrung vs. USD.
    Positive Differenz → Kapitalzufluss aus USD-Anlagen → bullish.
    Features: Rate-Spread vs. USD (direkt + im Kontext des Niveaus).
    """
    model_id     = "a2_zinsdifferenz"
    description  = "A2: Zinsdifferenz vs. USD (Carry-Trade-Signal)"
    feature_cols = [
        "a_rate_spread_vs_usd",
        "a_interest_rate",
    ]


class A3Zinserwartung(BaseForexModel):
    """
    A3 — Zinserwartung.

    Proxied ueber Kalender-Events: erwartete Zinsaenderung (forecast - aktuell).
    a_rate_expected_change_filled = 0 wenn kein Event in den naechsten 14 Tagen.
    a_rate_event_upcoming         = 1 wenn Event ansteht.

    Positiver expected_change + anstehendes Event → hawkish → bullish.
    """
    model_id     = "a3_zinserwartung"
    description  = "A3: Erwartete Zinsaenderung aus Kalender (forward-looking)"
    feature_cols = [
        "a_rate_expected_change_filled",
        "a_rate_event_upcoming",
        "a_rate_spread_vs_usd",
    ]


class A4ZBHaltung(BaseForexModel):
    """
    A4 — ZB Haltung (Hawk/Dove Proxy).

    Schätzt hawkishe/dovische Haltung der Zentralbank aus Positionsdaten.
    Abweichung über dem historischen Durchschnitt + steigend = hawkish.
    Features: Niveau, Abweichung, Spread (kombiniertes Bild der Zinspolitik).
    """
    model_id     = "a4_zb_haltung"
    description  = "A4: Hawk/Dove-Proxy aus Zinsposition und Trend"
    feature_cols = [
        "a_interest_rate",
        "a_rate_dev_from_avg",
        "a_rate_spread_vs_usd",
        "a_rate_hist_avg_3y",
    ]


class A5Zinsuberraschung(BaseForexModel):
    """
    A5 — Zinsüberraschung.

    Misst den Surprise-Effekt: grosse erwartete Aenderung =
    moegliche Ueberraschung wenn anders ausgeht.
    Kombination aus Erwartung und aktueller Abweichung vom Durchschnitt.
    """
    model_id     = "a5_zinsuberraschung"
    description  = "A5: Zins-Surprise — erwartete vs. historische Rate"
    feature_cols = [
        "a_rate_expected_change_filled",
        "a_rate_event_upcoming",
        "a_rate_dev_from_avg",
        "a_interest_rate",
    ]


# ── Modell-Registry ────────────────────────────────────────────────────────

GROUP_A_MODEL_CLASSES: list[type[BaseForexModel]] = [
    A1LeitzinsAbsolut,
    A2Zinsdifferenz,
    A3Zinserwartung,
    A4ZBHaltung,
    A5Zinsuberraschung,
]


def load_all_models(model_dir: Path = MODELS_DIR) -> dict[str, BaseForexModel]:
    """
    Laedt alle gespeicherten Gruppe-A-Modelle aus model_dir.
    Gibt Dict {model_id: model_instance} zurueck.
    Fehlende Modelle werden uebersprungen (mit Warnung).
    """
    models: dict[str, BaseForexModel] = {}
    for cls in GROUP_A_MODEL_CLASSES:
        path = model_dir / f"{cls.model_id}.joblib"
        if not path.exists():
            logger.warning("Modell nicht gefunden: %s", path)
            continue
        try:
            model = cls.load(path)
            models[cls.model_id] = model
        except Exception as exc:
            logger.error("Fehler beim Laden von %s: %s", path, exc)
    return models


def predict_all(
    df: pd.DataFrame,
    models: dict[str, BaseForexModel],
    latest_only: bool = True,
) -> pd.DataFrame:
    """
    Fuehrt alle geladenen Modelle auf dem DataFrame aus.

    Rueckgabe: DataFrame mit Spalten
      [currency, model_id, direction, confidence]
    """
    rows: list[dict] = []
    for model_id, model in models.items():
        if latest_only:
            preds = model.predict_latest(df)
        else:
            preds = model.predict(df)
        for p in preds:
            rows.append({
                "currency":   p.currency,
                "model_id":   p.model_id,
                "direction":  p.direction,
                "confidence": round(p.confidence, 1),
            })
    if not rows:
        return pd.DataFrame(columns=["currency", "model_id", "direction", "confidence"])
    return pd.DataFrame(rows).sort_values(["currency", "model_id"]).reset_index(drop=True)
