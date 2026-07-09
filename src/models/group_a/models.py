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

try:
    from xgboost import XGBClassifier as _XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

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
    currency:    str
    direction:   str   # "BULLISH" | "NEUTRAL" | "BEARISH"
    confidence:  float # 0–100
    model_id:    str
    bull_proba:  float = 0.0  # P(↑) raw probability 0–1
    bear_proba:  float = 0.0  # P(↓) raw probability 0–1


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
    n_refit:               int = 0   # Samples beim finalen Refit auf 100% der Daten (nach OOS-Eval)


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

            class_list = list(classes)
            bull_p = float(proba[class_list.index(1)])  if 1  in class_list else 0.0
            bear_p = float(proba[class_list.index(-1)]) if -1 in class_list else 0.0

            results.append(ModelResult(
                currency   = str(row.get("currency", "UNKNOWN")),
                direction  = direction,
                confidence = confidence,
                model_id   = self.model_id,
                bull_proba = bull_p,
                bear_proba = bear_p,
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


# ── XGBoost Basis-Klasse ───────────────────────────────────────────────────

XGB_PARAMS: dict = {
    "n_estimators":    200,
    "max_depth":         4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "eval_metric":   "mlogloss",
    "random_state":     42,
    "n_jobs":           -1,
}


class _XGBLabelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper um XGBClassifier: mappt beliebige Labels auf [0, n_classes-1].
    XGBoost akzeptiert nativ keine negativen Labels (-1, 0, 1).
    """

    def __init__(self, **kwargs):
        self._xgb_params = kwargs
        self._xgb: "_XGBClassifier | None" = None
        self._le = LabelEncoder()

    def fit(self, X, y, sample_weight=None):
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost nicht installiert: pip install xgboost")
        y_enc = self._le.fit_transform(y)
        self._xgb = _XGBClassifier(**self._xgb_params)
        self._xgb.fit(X, y_enc, sample_weight=sample_weight)
        self.classes_ = self._le.classes_        # sklearn-Konvention
        return self

    def predict(self, X) -> np.ndarray:
        y_enc = self._xgb.predict(X)
        return self._le.inverse_transform(y_enc.astype(int))

    def predict_proba(self, X) -> np.ndarray:
        return self._xgb.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._xgb.feature_importances_


class BaseForexModelXGB(BaseForexModel):
    """XGBoost-Variante von BaseForexModel mit Label-Encoding für {-1,0,1}."""

    def _build_pipeline(self) -> Pipeline:
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost nicht installiert: pip install xgboost")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    _XGBLabelWrapper(**XGB_PARAMS)),
        ])


class BaseForexModelXGBSelectK(BaseForexModelXGB):
    """XGBoost + SelectKBest(k) für erweiterte Feature-Sets — reduziert Rauschen."""
    select_k: int = 5

    def _build_pipeline(self) -> Pipeline:
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost nicht installiert: pip install xgboost")
        k = min(self.select_k, len(self.feature_cols))
        return Pipeline([
            ("scaler",   StandardScaler()),
            ("selector", SelectKBest(f_classif, k=k)),
            ("clf",      _XGBLabelWrapper(**XGB_PARAMS)),
        ])

    def fit(self, train_df: pd.DataFrame, target_col: str = "_target") -> dict:
        X, y = self._clean_data(train_df, target_col)
        if len(X) < 30:
            raise ValueError(f"{self.model_id}: Zu wenige Samples: {len(X)}")

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)

        selector = self._pipeline.named_steps["selector"]
        clf      = self._pipeline.named_steps["clf"]
        selected = [c for c, ok in zip(self.feature_cols, selector.get_support()) if ok]
        self._feature_importance = dict(zip(selected, clf.feature_importances_))
        self._is_fitted = True

        preds = self._pipeline.predict(X)
        train_acc = float(np.mean(preds == y))
        logger.info(
            "%s trainiert — %d/%d Features, Train-Acc: %.3f",
            self.model_id, len(selected), len(self.feature_cols), train_acc,
        )
        return {"n_train": len(X), "train_accuracy": train_acc}


class BaseForexModelXGBCurrency(BaseForexModelXGB):
    """XGBoost mit Currency-Gewichtung: Zeilen der Ziel-Währung erhalten höheres Sample-Weight."""
    weighted_currencies: list = []   # z.B. ["CAD"] oder ["JPY"]
    currency_weight: float = 3.0     # Gewichtungsfaktor für Ziel-Währung

    def fit(self, train_df: pd.DataFrame, target_col: str = "_target") -> dict:
        cols  = self.feature_cols + [target_col]
        valid = (
            train_df[cols + ["currency"]].dropna(subset=cols)
            if "currency" in train_df.columns
            else train_df[cols].dropna()
        )
        X = valid[self.feature_cols].to_numpy(dtype=float)
        y = valid[target_col].to_numpy(dtype=int)
        if len(X) < 30:
            raise ValueError(f"{self.model_id}: Zu wenige Samples: {len(X)}")

        weights = np.ones(len(X))
        if self.weighted_currencies and "currency" in valid.columns:
            mask = valid["currency"].isin(self.weighted_currencies).values
            weights[mask] = self.currency_weight

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y, clf__sample_weight=weights)

        clf = self._pipeline.named_steps["clf"]
        self._feature_importance = dict(zip(self.feature_cols, clf.feature_importances_))
        self._is_fitted = True

        preds = self._pipeline.predict(X)
        train_acc = float(np.mean(preds == y))
        logger.info(
            "%s trainiert — %d Samples (w=%.1f für %s), Train-Acc: %.3f",
            self.model_id, len(X), self.currency_weight,
            self.weighted_currencies, train_acc,
        )
        return {"n_train": len(X), "train_accuracy": train_acc}


# ── Ensemble-Infrastruktur ─────────────────────────────────────────────────

class _EnsemblePipeline:
    """
    Fake-Pipeline für den WF-Backtester: kombiniert mehrere gefittete
    sklearn-Pipelines per Majority- oder Weighted-Voting.

    predict(X_full) erwartet X mit allen Union-Features in der Reihenfolge
    von all_feature_cols; extrahiert pro Member den richtigen Sub-Slice.
    """

    def __init__(
        self,
        pipelines:       list,
        member_cols:     list[list[str]],
        all_feature_cols: list[str],
        weights:         list | None,
        strategy:        str,
    ) -> None:
        self.pipelines   = pipelines
        self.member_cols = member_cols
        self.strategy    = strategy
        self.weights     = weights
        self._idx        = {c: i for i, c in enumerate(all_feature_cols)}

    def predict(self, X_full: np.ndarray) -> np.ndarray:
        preds = []
        for pipe, cols in zip(self.pipelines, self.member_cols):
            idx = [self._idx[c] for c in cols]
            preds.append(pipe.predict(X_full[:, idx]))

        arr = np.array(preds, dtype=float)   # (n_members, n_samples)

        if self.strategy == "majority":
            # Labels {-1,+1}: sum > 0  ↔  ≥3/5 stimmen für BULL
            return np.where(arr.sum(axis=0) > 0, 1, -1).astype(int)

        # "weighted": accuracy-gewichtetes Voting
        w = np.array(self.weights if self.weights else [1.0] * len(self.pipelines))
        w = w[: len(self.pipelines)]
        w = w / w.sum()
        return np.where((arr * w[:, None]).sum(axis=0) > 0, 1, -1).astype(int)


class EnsembleForexModel(BaseForexModel):
    """
    Ensemble mehrerer Forex-Modelle mit Majority- oder Weighted-Voting.

    Unterklassen setzen:
        member_classes    — Liste der Mitglieds-Modellklassen
        strategy          — "majority" | "weighted"
        accuracy_weights  — Genauigkeiten aus Runde 2 (für strategy="weighted")
    """

    member_classes:   list       = []
    strategy:         str        = "majority"
    accuracy_weights: list | None = None

    def __init__(self) -> None:
        super().__init__()
        # feature_cols = geordnete Union aller Member-Feature-Sets
        seen: dict = {}
        for cls in self.member_classes:
            for col in cls.feature_cols:
                seen.setdefault(col, None)
        self.feature_cols = list(seen.keys())

    def _build_pipeline(self):
        return None  # wird nicht direkt genutzt

    def fit(self, train_df: pd.DataFrame, target_col: str = "_target") -> dict:
        fitted_pipes: list = []
        fitted_cols:  list = []

        for cls in self.member_classes:
            member = cls()
            try:
                member.fit(train_df, target_col)
                fitted_pipes.append(member._pipeline)
                fitted_cols.append(list(member.feature_cols))
            except (ValueError, RuntimeError) as exc:
                logger.debug("Ensemble-Member %s übersprungen: %s", cls.model_id, exc)

        if not fitted_pipes:
            raise ValueError(
                f"{self.model_id}: Alle {len(self.member_classes)} Mitglieder fehlgeschlagen"
            )

        weights = (
            list(self.accuracy_weights[: len(fitted_pipes)])
            if self.accuracy_weights else None
        )
        self._pipeline = _EnsemblePipeline(
            fitted_pipes, fitted_cols, self.feature_cols, weights, self.strategy
        )
        self._is_fitted = True

        # Train-Accuracy auf Union-Features (NaN-bereinigt)
        valid_cols = [c for c in self.feature_cols + [target_col] if c in train_df.columns]
        valid      = train_df[valid_cols].dropna()
        n_train    = len(valid)
        train_acc  = float("nan")
        if n_train >= 10 and all(c in valid.columns for c in self.feature_cols):
            X_tr      = valid[self.feature_cols].to_numpy(dtype=float)
            y_tr      = valid[target_col].to_numpy(dtype=int)
            train_acc = float(np.mean(self._pipeline.predict(X_tr) == y_tr))

        logger.info(
            "%s trainiert — %d/%d Mitglieder aktiv, Train-Acc: %s",
            self.model_id, len(fitted_pipes), len(self.member_classes),
            f"{train_acc:.3f}" if not np.isnan(train_acc) else "n/a",
        )
        return {"n_train": n_train, "train_accuracy": train_acc}


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
    # A4ZBHaltung entfernt: 87.1% Korrelation mit A1, schwaecher (50.8% vs 51.8%)
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
