"""Integration tests for FastAPI REST API endpoints."""

import io
import base64
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "model_loaded" in data


def test_api_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "encoder_backbone" in data
    assert "decoder_dim" in data


def test_api_predict_multipart_image():
    # Create sample in-memory image
    img = Image.new("RGB", (224, 224), color=(50, 100, 150))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")},
        data={"method": "greedy", "max_len": 15}
    )
    assert response.status_code == 200
    data = response.json()
    assert "caption" in data
    assert "tokens" in data
    assert "latency_ms" in data
    assert data["method"] == "greedy"


def test_api_predict_base64_payload():
    img = Image.new("RGB", (100, 100), color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "image_base64": b64_str,
        "method": "beam",
        "beam_width": 3,
        "max_len": 10
    }
    response = client.post("/predict-base64", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "caption" in data
    assert data["confidence_score"] is not None
