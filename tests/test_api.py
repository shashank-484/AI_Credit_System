import pytest

from app import audit_store
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(audit_store, "DB_PATH", tmp_path / "audit.sqlite")
    audit_store.initialize()
    return TestClient(app)


def test_summary_and_schema_load_trained_model(client):
    summary = client.get("/api/summary")
    assert summary.status_code == 200
    assert summary.json()["dataset"].startswith("Statlog (German Credit Data)")
    schema = client.get("/api/schema")
    assert schema.status_code == 200
    assert len(schema.json()["fields"]) == 20


def test_prediction_is_audited_and_review_chain_verifies(client):
    example = client.get("/api/examples").json()["typical_profile"]
    prediction = client.post("/api/predict", json={"values": example})
    assert prediction.status_code == 200
    result = prediction.json()
    assert 0 <= result["estimated_bad_risk_probability"] <= 1
    assert result["decision_id"]
    review = client.post(
        f"/api/audit/{result['decision_id']}/review",
        json={"action": "review_requested", "note": "Classroom review example"},
    )
    assert review.status_code == 200
    assert client.get("/api/audit/integrity").json()["valid"] is True


def test_invalid_category_is_rejected(client):
    example = client.get("/api/examples").json()["typical_profile"]
    example["checking_status"] = "not-a-valid-code"
    response = client.post("/api/predict", json={"values": example})
    assert response.status_code == 422
