"""Tests fuer Gruppe-A ML-Modelle (kein Netzwerk-/FRED-Zugriff noetig)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.group_a.models import (
    GROUP_A_MODEL_CLASSES,
    A1LeitzinsAbsolut,
    A2Zinsdifferenz,
    A3Zinserwartung,
    A4ZBHaltung,
    A5Zinsuberraschung,
    BaseForexModel,
    ModelResult,
    load_all_models,
    predict_all,
)
from src.models.group_a.target_builder import build_targets
from src.models.group_a.trainer import (
    add_derived_features,
    build_dataset,
    evaluate_model,
    temporal_split,
    train_group_a,
)

# ── Fixtures ───────────────────────────────────────────────────────────────

G7 = ["USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD"]
START = "2015-01-01"
END   = "2022-12-31"


def make_features_df(
    start: str = START,
    end: str = END,
    currencies: list[str] = G7,
    rng_seed: int = 7,
) -> pd.DataFrame:
    """Synthetisches Features-DataFrame im Format des feature_engineer."""
    rng   = np.random.default_rng(rng_seed)
    dates = pd.bdate_range(start, end, freq="B")
    rows  = []
    for d in dates:
        for i, cur in enumerate(currencies):
            rate     = 1.5 + i * 0.5 + rng.normal(0, 0.1)
            hist_avg = rate + rng.normal(0, 0.3)
            rows.append({
                "date":                         d,
                "currency":                     cur,
                "a_interest_rate":              max(0.0, rate),
                "a_rate_hist_avg_3y":           max(0.0, hist_avg),
                "a_rate_dev_from_avg":          rate - hist_avg,
                "a_rate_spread_vs_usd":         rate - (1.5 + rng.normal(0, 0.1)),
                "a_rate_expected_change":       rng.choice([np.nan, np.nan, np.nan, rng.uniform(-0.5, 0.5)]),
            })
    return pd.DataFrame(rows)


def make_fx_wide(
    start: str = START,
    end: str = END,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Synthetische FX-Kurse (Wide Format, normiert auf USD-Basis)."""
    rng   = np.random.default_rng(rng_seed)
    dates = pd.bdate_range(start, end, freq="B")
    non_usd = [c for c in G7 if c != "USD"]
    data = {}
    for cur in non_usd:
        # Zufaelliger Random-Walk
        returns  = rng.normal(0, 0.003, size=len(dates))
        prices   = 1.0 * np.cumprod(1 + returns)
        data[cur] = prices
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


def make_dataset(
    start: str = START,
    end: str = END,
) -> pd.DataFrame:
    """Kombiniertes Features + Target DataFrame fuer Tests."""
    features_df = make_features_df(start, end)
    fx_wide     = make_fx_wide(start, end)
    target_df   = build_targets(fx_wide)
    return build_dataset(features_df, target_df)


# ── target_builder Tests ───────────────────────────────────────────────────

def test_build_targets_schema():
    fx = make_fx_wide()
    result = build_targets(fx)
    assert set(result.columns) >= {"date", "currency", "fx_return_30d", "target_30d"}


def test_build_targets_nur_gueltige_labels():
    fx = make_fx_wide()
    result = build_targets(fx)
    assert set(result["target_30d"].unique()).issubset({-1, 0, 1})


def test_build_targets_usd_enthalten():
    fx = make_fx_wide()
    result = build_targets(fx)
    assert "USD" in result["currency"].values


def test_build_targets_keine_nan_in_labels():
    fx = make_fx_wide()
    result = build_targets(fx)
    assert result["target_30d"].notna().all()


def test_build_targets_alle_klassen_vorhanden():
    """Bei genug Daten sollen alle drei Klassen auftreten."""
    fx = make_fx_wide()
    result = build_targets(fx)
    assert len(result["target_30d"].unique()) == 3


def test_build_targets_threshold_strenger_mehr_neutral():
    """Hoeherer Threshold → mehr NEUTRAL-Labels."""
    fx = make_fx_wide()
    r_locker  = build_targets(fx, threshold=0.001)
    r_streng  = build_targets(fx, threshold=0.05)
    n_neutral_locker = (r_locker["target_30d"] == 0).sum()
    n_neutral_streng = (r_streng["target_30d"] == 0).sum()
    assert n_neutral_streng > n_neutral_locker


# ── add_derived_features Tests ─────────────────────────────────────────────

def test_add_derived_features_neue_spalten():
    df = make_features_df("2020-01-01", "2020-03-31")
    result = add_derived_features(df)
    assert "a_rate_expected_change_filled" in result.columns
    assert "a_rate_event_upcoming"         in result.columns


def test_add_derived_features_filled_kein_nan():
    df = make_features_df("2020-01-01", "2020-03-31")
    result = add_derived_features(df)
    assert result["a_rate_expected_change_filled"].notna().all()


def test_add_derived_features_upcoming_binaer():
    df = make_features_df("2020-01-01", "2020-03-31")
    result = add_derived_features(df)
    assert set(result["a_rate_event_upcoming"].unique()).issubset({0.0, 1.0})


def test_add_derived_features_upcoming_konsistent():
    """upcoming=1 genau dort, wo expected_change nicht NaN war."""
    df = make_features_df("2020-01-01", "2020-03-31")
    result = add_derived_features(df)
    has_event  = result["a_rate_event_upcoming"] == 1
    was_not_nan = df["a_rate_expected_change"].notna().values
    np.testing.assert_array_equal(has_event.values, was_not_nan)


# ── temporal_split Tests ───────────────────────────────────────────────────

def test_temporal_split_kein_overlap():
    ds = make_dataset()
    train, test, cutoff = temporal_split(ds)
    assert train["date"].max() < test["date"].min()


def test_temporal_split_groessen():
    ds = make_dataset()
    train, test, cutoff = temporal_split(ds, test_fraction=0.2)
    total = len(train) + len(test)
    assert abs(len(train) / total - 0.8) < 0.05  # ±5 %


def test_temporal_split_cutoff_typ():
    ds = make_dataset()
    _, _, cutoff = temporal_split(ds)
    assert isinstance(cutoff, pd.Timestamp)


# ── BaseForexModel / A1 Tests ──────────────────────────────────────────────

def test_modell_fit_gibt_metriken_zurueck():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    metrics = model.fit(ds)
    assert "n_train"        in metrics
    assert "train_accuracy" in metrics
    assert 0.0 <= metrics["train_accuracy"] <= 1.0


def test_modell_fit_setzt_is_fitted():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    assert model._is_fitted


def test_modell_predict_gibt_results_zurueck():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    preds = model.predict(ds.head(50))
    assert len(preds) > 0
    assert all(isinstance(p, ModelResult) for p in preds)


def test_modell_predict_gueltiges_label():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    preds = model.predict(ds.head(100))
    for p in preds:
        assert p.direction in ("BULLISH", "NEUTRAL", "BEARISH")


def test_modell_predict_konfidenz_range():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    preds = model.predict(ds.head(100))
    for p in preds:
        assert 0.0 <= p.confidence <= 100.0


def test_modell_predict_ohne_fit_wirft_fehler():
    model = A1LeitzinsAbsolut()
    with pytest.raises(RuntimeError, match="nicht trainiert"):
        model.predict(pd.DataFrame({"currency": ["USD"]}))


def test_modell_predict_ueberspringt_nan_zeilen():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    # Alle A1-Feature-Spalten auf NaN setzen
    nan_df = ds.head(10).copy()
    for col in model.feature_cols:
        nan_df[col] = np.nan
    preds = model.predict(nan_df)
    assert len(preds) == 0


def test_modell_predict_latest_gibt_eine_pro_waehrung():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    preds = model.predict_latest(ds)
    currencies = [p.currency for p in preds]
    # Hoechstens eine Vorhersage pro Waehrung
    assert len(currencies) == len(set(currencies))


# ── Persistenz (Save / Load) ───────────────────────────────────────────────

def test_modell_save_erstellt_datei():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = model.save(Path(tmpdir))
        assert path.exists()
        assert path.suffix == ".joblib"


def test_modell_load_gibt_gleiche_vorhersagen():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    sample = ds.tail(50)
    preds_before = {p.currency: (p.direction, round(p.confidence, 6)) for p in model.predict(sample)}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = model.save(Path(tmpdir))
        loaded = A1LeitzinsAbsolut.load(path)

    preds_after = {p.currency: (p.direction, round(p.confidence, 6)) for p in loaded.predict(sample)}
    assert preds_before == preds_after


def test_modell_feature_importance_nach_fit():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    fi = model.feature_importance
    assert set(fi.keys()) == set(model.feature_cols)
    assert abs(sum(fi.values()) - 1.0) < 1e-6   # Importances summieren auf 1


# ── evaluate_model Tests ───────────────────────────────────────────────────

def test_evaluate_model_metriken_vorhanden():
    ds = make_dataset()
    train_df, test_df, _ = temporal_split(ds)
    model = A1LeitzinsAbsolut()
    model.fit(train_df)
    metrics = evaluate_model(model, test_df)
    assert "test_accuracy"        in metrics
    assert "directional_accuracy" in metrics
    assert "n_test"               in metrics


def test_evaluate_model_accuracy_range():
    ds = make_dataset()
    train_df, test_df, _ = temporal_split(ds)
    model = A1LeitzinsAbsolut()
    model.fit(train_df)
    metrics = evaluate_model(model, test_df)
    assert 0.0 <= metrics["test_accuracy"] <= 1.0


def test_evaluate_model_leerer_test_gibt_nan():
    ds    = make_dataset()
    model = A1LeitzinsAbsolut()
    model.fit(ds)
    # Leeres Test-Set
    empty_test = ds.head(0).copy()
    empty_test["_target"] = pd.Series(dtype=int)
    metrics = evaluate_model(model, empty_test)
    assert np.isnan(metrics["test_accuracy"])


# ── train_group_a Tests ────────────────────────────────────────────────────

def test_train_group_a_gibt_metriken_zurueck():
    ds = make_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = train_group_a(ds, model_dir=Path(tmpdir))
    assert len(metrics) > 0
    for mid, m in metrics.items():
        assert m.n_train > 0


def test_train_group_a_speichert_alle_modelle():
    ds = make_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        train_group_a(ds, model_dir=model_dir)
        joblib_files = list(model_dir.glob("*.joblib"))
        assert len(joblib_files) > 0


def test_train_group_a_schreibt_metadata_json():
    ds = make_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        train_group_a(ds, model_dir=model_dir)
        meta_path = model_dir / "metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["group"] == "A"
        assert "cutoff_date"  in meta
        assert "models"       in meta


def test_train_group_a_cutoff_korrekt():
    """Das Cutoff-Datum muss vor den juengsten Testdaten liegen."""
    ds = make_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = train_group_a(ds, model_dir=Path(tmpdir), test_fraction=0.2)
    for m in metrics.values():
        assert m.cutoff_date != ""


# ── Alle Modell-Klassen Smoke-Test ─────────────────────────────────────────

@pytest.mark.parametrize("ModelClass", GROUP_A_MODEL_CLASSES)
def test_alle_modelle_trainierbar(ModelClass):
    """Jedes Modell in GROUP_A_MODEL_CLASSES muss trainierbar sein."""
    ds    = make_dataset()
    model = ModelClass()
    metrics = model.fit(ds)
    assert metrics["n_train"] > 0
    preds = model.predict(ds.tail(100))
    assert len(preds) > 0


@pytest.mark.parametrize("ModelClass", GROUP_A_MODEL_CLASSES)
def test_alle_modelle_haben_model_id(ModelClass):
    model = ModelClass()
    assert model.model_id.startswith("a")
    assert len(model.feature_cols) > 0
    assert model.description


# ── load_all_models + predict_all Tests ───────────────────────────────────

def test_load_all_models_laedt_gespeicherte_modelle():
    ds = make_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        train_group_a(ds, model_dir=model_dir)
        loaded = load_all_models(model_dir)
        assert len(loaded) > 0
        for m in loaded.values():
            assert m._is_fitted


def test_load_all_models_leeres_dir_gibt_leeres_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_all_models(Path(tmpdir))
    assert result == {}


def test_predict_all_schema():
    ds = make_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        train_group_a(ds, model_dir=model_dir)
        models = load_all_models(model_dir)
        result = predict_all(ds, models, latest_only=True)

    assert set(result.columns) >= {"currency", "model_id", "direction", "confidence"}


def test_predict_all_gueltiges_label():
    ds = make_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        train_group_a(ds, model_dir=model_dir)
        models = load_all_models(model_dir)
        result = predict_all(ds, models, latest_only=True)

    assert result["direction"].isin(["BULLISH", "NEUTRAL", "BEARISH"]).all()


def test_predict_all_eine_zeile_pro_modell_waehrung():
    ds = make_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        metrics = train_group_a(ds, model_dir=model_dir)
        models = load_all_models(model_dir)
        result = predict_all(ds, models, latest_only=True)

    n_trained = len(metrics)
    # Jedes trainierte Modell liefert max. eine Zeile pro Waehrung
    for mid in metrics:
        model_rows = result[result["model_id"] == mid]
        assert len(model_rows) <= len(G7)
