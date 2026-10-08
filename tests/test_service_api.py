from __future__ import annotations

from fastapi.testclient import TestClient

from solarvision.generate import encode_base64_png
from solarvision.labels import DefectClass
from solarvision.main import app
from tests.conftest import make_panel

client = TestClient(app)


def test_assess_base64_image():
    img = make_panel(DefectClass.HOTSPOT, seed=2)
    r = client.post("/assess", json={"image_base64": encode_base64_png(img)})
    assert r.status_code == 200
    body = r.json()["assessment"]
    assert body["predicted_class"] == "hotspot"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["decided_by"] in {"rule", "model"}


def test_assess_generated_sample():
    r = client.post(
        "/assess",
        json={"sample": {"defect": "dead_cell", "seed": 4, "severity": 0.85, "size_frac": 0.7}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ground_truth"] == "dead_cell"
    assert body["correct"] is True
    assert body["image_base64"]
    assert body["assessment"]["predicted_class"] == "dead_cell"


def test_assess_requires_exactly_one_input():
    # Neither field.
    assert client.post("/assess", json={}).status_code == 422
    # Both fields.
    img = encode_base64_png(make_panel(DefectClass.HEALTHY, seed=1))
    both = client.post("/assess", json={"image_base64": img, "sample": {"defect": "healthy"}})
    assert both.status_code == 422


def test_unknown_defect_is_rejected():
    r = client.post("/assess", json={"sample": {"defect": "melted"}})
    assert r.status_code == 422


def test_malformed_base64_is_rejected():
    r = client.post("/assess", json={"image_base64": "not-a-real-png"})
    assert r.status_code == 422


def test_classes_endpoint():
    body = client.get("/classes").json()
    assert body["classes"] == ["healthy", "hotspot", "crack", "dead_cell"]
    assert "healthy" not in body["defects"]
