"""
Tests for API key authentication.
"""

import pytest
from fastapi.testclient import TestClient


def test_missing_api_key_returns_401(client):
    """Requests without X-API-Key header should be rejected."""
    # Use the wrong key to isolate auth from other errors
    response = client.post(
        "/detect/image/",
        headers={"X-API-Key": ""},  # empty key
        files={"file": ("test.png", b"fake", "image/png")}
    )
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"

def test_wrong_api_key_returns_401(client):
    """
    Requests with an invalid API key should be rejected.
    """
    response = client.post(
        "/detect/image?",
        headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"

def test_valid_api_key_passes(client, sample_image_bytes):
    """Requests with a valid API key should not return 401."""
    # client fixture already sets X-API-Key: test-key-123
    response = client.post(
        "/detect/image/",
        files={"file": ("test.png", sample_image_bytes, "image/png")}
    )
    assert response.status_code != 401


def test_health_endpoint_is_public(client):
    """Health endpoint should work without an API key."""
    response = client.get("/health", headers={})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"