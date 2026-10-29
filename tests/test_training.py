import numpy as np

from app.dataset import CATEGORY_LABELS
from app.training import build_schema, credit_error_cost, metric_report
from app.dataset import load_dataset


def test_cost_matrix_penalizes_missed_bad_risk_five_times_more():
    # One good case is incorrectly called bad (cost 1); one bad case is missed (cost 5).
    assert credit_error_cost(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == 1.5


def test_report_includes_majority_baseline_and_bad_recall():
    report = metric_report(np.array([0, 0, 0, 1]), np.array([0.1, 0.8, 0.4, 0.7]), 0.5)
    assert report["accuracy"] == 0.75
    assert report["majority_baseline_accuracy"] == 0.75
    assert report["bad_risk_recall"] == 1.0
    assert "expected_error_cost_per_case" in report


def test_schema_contains_corrected_complete_category_mappings():
    X, _ = load_dataset()
    schema = {field["name"]: field for field in build_schema(X)}
    checking = {choice["value"]: choice["label"] for choice in schema["checking_status"]["choices"]}
    assert checking["A11"] == CATEGORY_LABELS["checking_status"]["A11"]
    assert checking["A14"] == "No checking account"
    personal = {choice["value"] for choice in schema["personal_status_sex"]["choices"]}
    assert "A95" in personal
    assert schema["installment_rate"]["type"] == "ordered"
    assert schema["residence_since"]["type"] == "ordered"
