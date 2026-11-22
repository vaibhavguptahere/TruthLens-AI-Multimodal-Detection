"""
Integration tests for Phase 8: FastAPI Backend Endpoints.
"""

from pathlib import Path
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.api.main import app

client = TestClient(app)


def test_api_root_and_health():
    """Verify API root and health check endpoints respond with 200 OK."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"


def test_api_verify_endpoint_success():
    """Verify /api/verify processes an image and caption and returns a complete verification report."""
    # Create test image in memory
    img = Image.new("RGB", (100, 100), color=(120, 180, 240))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    files = {"image": ("test.jpg", buffer, "image/jpeg")}
    data = {"caption": "Official photograph of university campus during clear morning."}

    response = client.post("/api/verify", files=files, data=data)
    assert response.status_code == 200

    payload = response.json()
    assert payload["prediction"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert "confidence" in payload
    assert "consistency" in payload
    assert "token_attributions" in payload
    assert "gradcam_heatmap_base64" in payload
    assert "report" in payload
    assert payload["report"]["final_verdict"] == payload["prediction"]


def test_api_verify_empty_caption_error():
    """Verify /api/verify rejects requests with empty captions."""
    img = Image.new("RGB", (50, 50))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    files = {"image": ("test.jpg", buffer, "image/jpeg")}
    data = {"caption": "   "}

    response = client.post("/api/verify", files=files, data=data)
    assert response.status_code == 400
    assert "Text caption must not be empty" in response.json()["detail"]
