"""Responsible Credit AI prototype: predictions, review, audit, fairness, and monitoring APIs."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
import math

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import audit_store
from .dataset import CATEGORY_LABELS, DATA_PATH, display_value

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"

app = FastAPI(
    title="Responsible Credit AI",
    description="Educational prototype using the UCI Statlog German Credit benchmark. Not a real lending system.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
audit_store.initialize()


class PredictionRequest(BaseModel):
    values: dict[str, Any]


class ReviewRequest(BaseModel):
    action: Literal["review_requested", "review_upheld", "review_revised"]
    note: str = Field(min_length=5, max_length=1000)


@lru_cache(maxsize=1)
def _load_bundle_for_file(modified_ns: int, file_size: int) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run `python -m app.train_model` first.")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The trained model file could not be loaded; retrain the model.") from exc


def load_bundle() -> dict[str, Any]:
    """Cache the model, but load a new version after notebook/script retraining."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run `python -m app.train_model` first.")
    stat = MODEL_PATH.stat()
    return _load_bundle_for_file(stat.st_mtime_ns, stat.st_size)


def validate_values(values: dict[str, Any], schema: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {field["name"] for field in schema}
    missing, extra = sorted(expected - set(values)), sorted(set(values) - expected)
    if missing or extra:
        raise HTTPException(status_code=422, detail={"missing_fields": missing, "unexpected_fields": extra})
    normalized: dict[str, Any] = {}
    for field in schema:
        name, value = field["name"], values[field["name"]]
        try:
            if field["type"] == "category":
                normalized[name] = str(value)
                allowed = {str(choice["value"]) for choice in field["choices"]}
                if normalized[name] not in allowed:
                    raise ValueError
            elif field["type"] == "ordered":
                normalized[name] = int(value)
                allowed = {int(choice["value"]) for choice in field["choices"]}
                if normalized[name] not in allowed:
                    raise ValueError
            else:
                numeric = float(value)
                if not math.isfinite(numeric) or not numeric.is_integer():
                    raise ValueError
                if not field["min"] <= numeric <= field["max"]:
                    raise ValueError
                normalized[name] = int(numeric)
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(status_code=422, detail=f"Invalid value for {field['label']}; use one of the listed choices or the shown numeric range.") from None
    return normalized


def predict_probability(bundle: dict[str, Any], values: dict[str, Any]) -> float:
    row = pd.DataFrame([{name: values[name] for name in bundle["feature_names"]}])
    return float(bundle["pipeline"].predict_proba(row)[0, 1])


def local_sensitivity(bundle: dict[str, Any], values: dict[str, Any], probability: float) -> list[dict[str, Any]]:
    """One-feature reference substitutions; useful what-if signals, not causal explanations."""
    explanations = []
    for field in bundle["feature_names"]:
        reference = bundle["reference_values"][field]
        if values[field] == reference:
            continue
        changed = dict(values)
        changed[field] = reference
        reference_probability = predict_probability(bundle, changed)
        delta = probability - reference_probability
        explanations.append({
            "field": field,
            "label": next((item["label"] for item in bundle["schema"] if item["name"] == field), field),
            "value": display_value(field, values[field]),
            "reference_value": display_value(field, reference),
            "probability_change": round(delta, 4),
            "direction": "raises" if delta > 0 else "lowers" if delta < 0 else "no measurable change",
        })
    explanations.sort(key=lambda item: abs(item["probability_change"]), reverse=True)
    return explanations[:5]


def drift_status(psi: float) -> str:
    if psi < 0.1:
        return "low"
    if psi < 0.25:
        return "moderate"
    return "high"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_trained": MODEL_PATH.exists(), "dataset_available": DATA_PATH.exists()}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    bundle = load_bundle()
    return {
        "project": "Regulatory-Aware AI for Explainable and Auditable Credit Decisions",
        "dataset": bundle["dataset"], "rows": bundle["dataset_rows"],
        "input_features": bundle["all_dataset_features"], "model_features": len(bundle["feature_names"]),
        "target": bundle["target_name"], "model_name": bundle["model_name"],
        "model_version": bundle["model_version"], "threshold": bundle["threshold"],
        "metrics": bundle["test_metrics"], "metrics_at_0_5": bundle["test_metrics_at_0_5"],
        "cv_comparison": bundle["cv_comparison"], "split": bundle["split"],
        "notice": "Historical benchmark for classroom use only. It does not make or recommend real loan decisions.",
    }


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    bundle = load_bundle()
    return {"fields": bundle["schema"], "reference_values": bundle["reference_values"]}


@app.get("/api/examples")
def examples() -> dict[str, Any]:
    bundle = load_bundle()
    return {"typical_profile": bundle["reference_values"], "description": "Training-set median/mode values; an illustrative profile, not an individual record."}


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    bundle = load_bundle()
    values = validate_values(request.values, bundle["schema"])
    probability = predict_probability(bundle, values)
    threshold = float(bundle["threshold"])
    predicted_bad = probability >= threshold
    explanations = local_sensitivity(bundle, values, probability)
    decision_id = audit_store.save_decision(values, probability, predicted_bad, threshold, bundle["model_version"], explanations)
    return {
        "decision_id": decision_id,
        "risk_label": "Bad-risk label predicted" if predicted_bad else "Good-risk label predicted",
        "predicted_bad_risk": predicted_bad,
        "estimated_bad_risk_probability": round(probability, 4),
        "threshold": round(threshold, 4),
        "model_name": bundle["model_name"], "model_version": bundle["model_version"],
        "feature_count": len(bundle["feature_names"]), "features_used": bundle["feature_names"],
        "explanations": explanations,
        "explanation_method": "One-feature reference sensitivity: replaces one input at a time with its training median/mode. It is an approximate what-if explanation, not causal evidence.",
        "notice": "Educational result from historical data; not a lending decision.",
    }


@app.get("/api/fairness")
def fairness() -> dict[str, Any]:
    bundle = load_bundle()
    return {
        "metrics": bundle["fairness"],
        "note": "Group metrics are descriptive on the held-out test split. Groups under 20 cases are flagged; this is not a legal fairness determination.",
    }


@app.get("/api/audit")
def audit(limit: int = 50) -> dict[str, Any]:
    return {"decisions": audit_store.recent_decisions(limit)}


@app.post("/api/audit/{decision_id}/review")
def review(decision_id: str, request: ReviewRequest) -> dict[str, str]:
    try:
        audit_store.add_review(decision_id, request.action, request.note.strip())
    except KeyError:
        raise HTTPException(status_code=404, detail="Decision record not found.") from None
    return {"status": "saved", "decision_id": decision_id, "action": request.action}


@app.get("/api/audit/integrity")
def audit_integrity() -> dict[str, Any]:
    return audit_store.verify_integrity()


@app.get("/api/monitoring")
def monitoring(limit: int = 100) -> dict[str, Any]:
    bundle = load_bundle()
    rows = audit_store.recent_inputs(limit)
    if len(rows) < 20:
        return {"status": "collecting", "records": len(rows), "minimum_records": 20, "features": [], "note": "Make at least 20 local demo predictions to compare recent inputs with the training distribution."}
    actual_frame = pd.DataFrame(rows)
    reference = bundle["drift_reference"]
    results = []
    eps = 1e-6
    for field, spec in reference["numeric"].items():
        actual = pd.to_numeric(actual_frame[field], errors="coerce").dropna().to_numpy(dtype=float)
        edges = np.asarray(spec["edges"], dtype=float)
        counts, _ = np.histogram(actual, bins=edges)
        observed = counts / max(counts.sum(), 1)
        expected = np.asarray(spec["proportions"], dtype=float)
        psi = float(np.sum((observed - expected) * np.log((observed + eps) / (expected + eps))))
        results.append({"field": field, "type": "numeric", "psi": round(psi, 4), "status": drift_status(psi)})
    for field, expected_map in reference["categorical"].items():
        actual = actual_frame[field].astype(str).value_counts(normalize=True).to_dict()
        categories = set(expected_map) | set(actual)
        psi = sum((actual.get(value, 0.0) - expected_map.get(value, 0.0)) * math.log((actual.get(value, 0.0) + eps) / (expected_map.get(value, 0.0) + eps)) for value in categories)
        results.append({"field": field, "type": "categorical", "psi": round(float(psi), 4), "status": drift_status(float(psi))})
    return {
        "status": "ready", "records": len(rows), "window": min(max(int(limit), 1), 500),
        "features": sorted(results, key=lambda row: row["psi"], reverse=True),
        "note": "Population Stability Index compares recent demo inputs with training data. It detects input distribution shifts, not accuracy or outcome drift; small windows are noisy.",
    }
