"""
Integration tests for API endpoints.
Uses real models — run with: pytest -m integration
"""
import pytest
import os
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def integration_client():
    """Real FastAPI client with models actually loaded."""
    os.environ["API_KEYS"] = "integration-test-key"
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "integration-test-key"}


class TestHealthEndpoint:

    def test_health_returns_ok(self, integration_client):
        response = integration_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "version" in response.json()


class TestImageEndpoint:

    def test_rejects_missing_api_key(self, integration_client, sample_image_bytes):
        response = integration_client.post(
            "/detect/image/",
            files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        assert response.status_code == 401

    def test_rejects_unsupported_mime_type(self, integration_client, auth_headers):
        response = integration_client.post(
            "/detect/image/",
            headers=auth_headers,
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 415

    def test_rejects_empty_file(self, integration_client, auth_headers):
        response = integration_client.post(
            "/detect/image/",
            headers=auth_headers,
            files={"file": ("test.png", b"", "image/png")}
        )
        assert response.status_code == 400

    def test_returns_valid_result_shape(self, integration_client, auth_headers, sample_image_bytes):
        response = integration_client.post(
            "/detect/image/",
            headers=auth_headers,
            files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "ai_probability" in data
        assert "confidence" in data
        assert "signals" in data
        assert "verdict" in data
        assert 0.0 <= data["ai_probability"] <= 1.0


class TestDocumentEndpoint:

    def test_returns_valid_result_shape(self, integration_client, auth_headers, sample_text_bytes):
        response = integration_client.post(
            "/detect/document/",
            headers=auth_headers,
            files={"file": ("test.txt", sample_text_bytes, "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "ai_probability" in data
        assert "word_count" in data
        assert "signals" in data
        assert 0.0 <= data["ai_probability"] <= 1.0