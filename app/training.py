"""Leakage-safe model comparison and evaluation for the UCI German Credit data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer

from .dataset import CATEGORY_LABELS, CATEGORICAL_FEATURES, DATA_PATH, FIELD_LABELS, NUMERIC_FEATURES, load_dataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "model_report.json"
RANDOM_STATE = 42
REDUCED_FEATURES = [
    "checking_status", "purpose", "duration", "installment_rate",
    "credit_history", "savings", "property", "job",
]


def credit_error_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cost matrix: bad risk called good costs 5; good risk called bad costs 1."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((fp + 5 * fn) / len(y_true))


def negative_credit_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -credit_error_cost(y_true, y_pred)


def make_pipeline(classifier: Any) -> Pipeline:
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numeric", numeric, NUMERIC_FEATURES),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def candidate_searches() -> list[tuple[str, list[str], Pipeline, dict[str, list[Any]]]]:
    """Predeclare candidate models and feature sets before touching the test set."""
    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE), {
            "classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"],
        }),
        ("Random Forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), {
            "classifier__n_estimators": [200], "classifier__max_depth": [None, 8],
            "classifier__min_samples_leaf": [1, 3], "classifier__max_features": ["sqrt"],
        }),
        ("Extra Trees", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1), {
            "classifier__n_estimators": [200], "classifier__max_depth": [None, 8],
            "classifier__min_samples_leaf": [1, 3], "classifier__max_features": ["sqrt"],
        }),
        ("Histogram Gradient Boosting", HistGradientBoostingClassifier(random_state=RANDOM_STATE), {
            "classifier__max_iter": [100, 200], "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_leaf_nodes": [15, 31],
        }),
    ]
    searches = []
    for feature_set_name, fields in [("All 20 fields", CATEGORICAL_FEATURES + NUMERIC_FEATURES), ("8-field reduced", REDUCED_FEATURES)]:
        categorical = [field for field in fields if field in CATEGORICAL_FEATURES]
        numeric = [field for field in fields if field in NUMERIC_FEATURES]
        for name, classifier, grid in candidates:
            categorical_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            numeric_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ])
            prep = ColumnTransformer([
                ("categorical", categorical_pipe, categorical),
                ("numeric", numeric_pipe, numeric),
            ], remainder="drop")
            pipeline = Pipeline([("preprocess", prep), ("classifier", classifier)])
            searches.append((f"{name} ({feature_set_name})", fields.copy(), pipeline, grid))
    return searches


def metric_report(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, predictions, labels=[0, 1]).ravel()
    majority = max(np.mean(y_true_array == 0), np.mean(y_true_array == 1))
    return {
        "accuracy": float(accuracy_score(y_true_array, predictions)),
        "majority_baseline_accuracy": float(majority),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, predictions)),
        "bad_risk_precision": float(precision_score(y_true_array, predictions, pos_label=1, zero_division=0)),
        "bad_risk_recall": float(recall_score(y_true_array, predictions, pos_label=1, zero_division=0)),
        "bad_risk_f1": float(f1_score(y_true_array, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, probabilities)),
        "brier_score": float(brier_score_loss(y_true_array, probabilities)),
        "expected_error_cost_per_case": credit_error_cost(y_true_array, predictions),
        "confusion_matrix_good_then_bad": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
        "test_rows": int(len(y_true_array)),
    }


def select_cost_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    costs = [credit_error_cost(y_true, probabilities >= threshold) for threshold in thresholds]
    minimum = min(costs)
    tied = [float(t) for t, cost in zip(thresholds, costs) if cost == minimum]
    threshold = min(tied, key=lambda t: abs(t - 0.5))
    return threshold, float(minimum)


def build_schema(X_train: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for name in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        if name in CATEGORICAL_FEATURES:
            options = sorted(CATEGORY_LABELS.get(name, {}).keys())
            default = str(X_train[name].mode().iloc[0])
            result.append({"name": name, "label": FIELD_LABELS[name], "type": "category", "choices": [{"value": value, "label": CATEGORY_LABELS[name].get(value, value)} for value in options], "default": default})
        elif name in {"installment_rate", "residence_since"}:
            values = sorted(int(value) for value in X_train[name].unique())
            result.append({"name": name, "label": FIELD_LABELS[name], "type": "ordered", "choices": [{"value": value, "label": CATEGORY_LABELS[name][str(value)]} for value in values], "default": int(X_train[name].mode().iloc[0])})
        else:
            values = pd.to_numeric(X_train[name], errors="coerce").dropna()
            result.append({
                "name": name, "label": FIELD_LABELS[name], "type": "number",
                "min": int(values.min()), "max": int(values.max()),
                "default": int(round(values.median())),
            })
    return result


def reference_values(X_train: pd.DataFrame) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in CATEGORICAL_FEATURES:
        values[name] = str(X_train[name].mode().iloc[0])
    for name in NUMERIC_FEATURES:
        values[name] = float(X_train[name].median())
    return values


def fairness_report(X: pd.DataFrame, y: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    frame = X.copy()
    frame["age_group"] = pd.cut(frame["age"], bins=[0, 25, 35, 50, np.inf], labels=["25 or younger", "26–35", "36–50", "Over 50"])
    result: dict[str, Any] = {}
    for attribute in ["personal_status_sex", "foreign_worker", "age_group"]:
        groups = []
        values = frame[attribute].astype(str).to_numpy()
        for group in sorted(set(values)):
            mask = values == group
            truth = y.to_numpy()[mask]
            pred = predictions[mask]
            positives = int(np.sum(truth == 1))
            negatives = int(np.sum(truth == 0))
            tp = int(np.sum((truth == 1) & (pred == 1)))
            fp = int(np.sum((truth == 0) & (pred == 1)))
            groups.append({
                "group": group,
                "n": int(mask.sum()),
                "predicted_bad_risk_rate": float(np.mean(pred)) if len(pred) else 0.0,
                "bad_risk_recall": float(tp / positives) if positives else None,
                "false_positive_rate": float(fp / negatives) if negatives else None,
                "small_sample_warning": bool(mask.sum() < 20),
            })
        result[attribute] = groups
    return result


def make_drift_reference(X_train: pd.DataFrame) -> dict[str, Any]:
    reference: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for field in NUMERIC_FEATURES:
        values = X_train[field].astype(float).to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 6))).tolist()
        if len(edges) < 2:
            edges = [float(values.min()) - 1, float(values.max()) + 1]
        edges[0] = float("-inf")
        edges[-1] = float("inf")
        counts, _ = np.histogram(values, bins=edges)
        reference["numeric"][field] = {"edges": edges, "proportions": (counts / max(counts.sum(), 1)).tolist()}
    for field in CATEGORICAL_FEATURES:
        proportions = X_train[field].astype(str).value_counts(normalize=True).to_dict()
        reference["categorical"][field] = {str(key): float(value) for key, value in proportions.items()}
    return reference


def main() -> None:
    X, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    scorers: dict[str, Any] = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "bad_risk_recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "negative_cost": make_scorer(negative_credit_cost),
    }
    comparisons: list[dict[str, Any]] = []
    fitted_candidates: dict[str, dict[str, Any]] = {}

    for name, fields, pipeline, grid in candidate_searches():
        print(f"Cross-validating {name}…")
        search = GridSearchCV(
            pipeline, grid, scoring=scorers, refit="accuracy", cv=cv,
            n_jobs=1, return_train_score=False, error_score="raise",
        )
        search.fit(x_train[fields], y_train)
        idx = int(search.best_index_)
        candidate_result = {
            "name": name,
            "fields": fields,
            "cv_accuracy_mean": float(search.cv_results_["mean_test_accuracy"][idx]),
            "cv_accuracy_std": float(search.cv_results_["std_test_accuracy"][idx]),
            "cv_balanced_accuracy_mean": float(search.cv_results_["mean_test_balanced_accuracy"][idx]),
            "cv_bad_risk_recall_mean": float(search.cv_results_["mean_test_bad_risk_recall"][idx]),
            "cv_expected_error_cost_mean": float(-search.cv_results_["mean_test_negative_cost"][idx]),
            "best_params": search.best_params_,
        }
        comparisons.append(candidate_result)
        fitted_candidates[name] = {"search": search, "fields": fields, "result": candidate_result}

    comparisons.sort(key=lambda row: (row["cv_accuracy_mean"], row["cv_balanced_accuracy_mean"]), reverse=True)
    winner_name = comparisons[0]["name"]
    winner = fitted_candidates[winner_name]
    best_pipeline: Pipeline = winner["search"].best_estimator_
    feature_names: list[str] = winner["fields"]

    # The threshold is selected from out-of-fold predictions on training data only.
    oof = cross_val_predict(
        best_pipeline, x_train[feature_names], y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1),
        method="predict_proba", n_jobs=1,
    )[:, 1]
    threshold, oof_cost = select_cost_threshold(y_train, oof)

    # GridSearchCV has refitted the selected pipeline on all training rows. The test set is now used once.
    test_probabilities = best_pipeline.predict_proba(x_test[feature_names])[:, 1]
    test_at_half = metric_report(y_test, test_probabilities, 0.5)
    test_policy = metric_report(y_test, test_probabilities, threshold)
    test_predictions = (test_probabilities >= threshold).astype(int)
    fairness = fairness_report(x_test, y_test, test_predictions)

    raw_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    config_text = json.dumps({"winner": winner_name, "params": winner["result"]["best_params"], "seed": RANDOM_STATE}, sort_keys=True)
    model_version = hashlib.sha256((raw_hash + config_text).encode("utf-8")).hexdigest()[:12]
    bundle = {
        "pipeline": best_pipeline,
        "feature_names": feature_names,
        "schema": build_schema(x_train),
        "reference_values": reference_values(x_train),
        "drift_reference": make_drift_reference(x_train),
        "threshold": threshold,
        "oof_cost_per_case": oof_cost,
        "model_name": winner_name,
        "model_version": model_version,
        "cv_comparison": comparisons,
        "test_metrics_at_0_5": test_at_half,
        "test_metrics": test_policy,
        "fairness": fairness,
        "dataset": "Statlog (German Credit Data), UCI dataset 144",
        "target_name": "credit risk: good (0) or bad (1)",
        "dataset_rows": int(len(X)),
        "all_dataset_features": int(X.shape[1]),
        "selected_features": feature_names,
        "random_state": RANDOM_STATE,
        "split": {"train_rows": int(len(x_train)), "test_rows": int(len(x_test)), "test_fraction": 0.2},
    }
    joblib.dump(bundle, MODEL_PATH)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in bundle.items() if key not in {"pipeline", "schema", "reference_values", "drift_reference"}}
    report["caveats"] = [
        "Only 1,000 historical cases; cross-validation and the held-out test estimate remain uncertain.",
        "Cost-aware threshold was selected using training out-of-fold predictions; held-out test data was not used for selection.",
        "Fairness group results with fewer than 20 held-out cases are marked as small samples.",
        "This benchmark estimates dataset labels and is not a real lending decision system.",
    ]
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nCross-validation comparison (selected candidate parameters):")
    for row in comparisons:
        print(f"  {row['name']}: accuracy={row['cv_accuracy_mean']:.3f} ± {row['cv_accuracy_std']:.3f}; bad recall={row['cv_bad_risk_recall_mean']:.3f}; cost={row['cv_expected_error_cost_mean']:.3f}")
    print(f"\nSelected: {winner_name} | features: {len(feature_names)} | cost-aware threshold: {threshold:.2f}")
    print("Held-out test metrics at policy threshold:")
    print(json.dumps(test_policy, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
