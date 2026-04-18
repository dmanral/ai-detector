"""
Shared test fixtures available to all tests.
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

#-----------------------------------------------------------------------------------------------------
# Pytest markers
#-----------------------------------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

#-----------------------------------------------------------------------------------------------------
# Sample file fixtures
#-----------------------------------------------------------------------------------------------------
@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    A small solid-color PNG - fast to create, no disk I/O.
    """
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """
    A small solid-color JPEG.
    """
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

@pytest.fixture
def sample_text_bytes() -> bytes:
    """
    Plain text document.
    """
    return b"This is a sample document for testing purposes. " * 20

@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """
    Minimal valid PDF with text content.
    """
    content = b"""%PDF-1.4
        1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
        2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
        3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
        /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
        4 0 obj << /Length 44 >> stream
        BT /F1 12 Tf 100 700 Td (Test document content.) Tj ET
        endstream endobj
        5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
        xref
        0 6
        0000000000 65535 f
        0000000009 00000 n
        0000000058 00000 n
        0000000115 00000 n
        0000000274 00000 n
        0000000369 00000 n
        trailer << /Size 6 /Root 1 0 R >>
        startxref
        441
        %%EOF"""
    return content


#-----------------------------------------------------------------------------------------------------
# Mock ML model fixtures
#-----------------------------------------------------------------------------------------------------
@pytest.fixture
def mock_image_service():
    """
    ImageDetectionService with all ML models mocked out.
    """
    from app.features.image.service import ImageDetectionService
    from app.features.image.schemas import (
        ImageDetectionResult,
        ImageSignals,
        ConfidenceLevel
    )

    service = ImageDetectionService()
    service._models_loaded = True
    service._ort_session = MagicMock()
    service._face_detector = MagicMock()
    service._landmark_predictor = MagicMock()
    service._classifier_labels = {"0": "artificial", "1": "real"}

    # Mock classifier to return AI score of 0.95.
    service._ort_session.run.return_value = [np.array([[2.0, -2.0]])]

    # Mock face detector to return no faces.
    service._face_detector.return_value = []

    return service

@pytest.fixture
def mock_document_service():
    """
    DocumentDetectionService with all ML models mocked out.
    """
    from app.features.document.service import DocumentDetectionService

    service = DocumentDetectionService()
    service._models_loaded = True
    service._ort_session = MagicMock()
    service._tokenizer = MagicMock()

    # Mock tokenizer to return dummy tensons.
    service._tokenizer.return_value = {
        "input_ids": np.zeros((1, 512), dtype=np.int64),
        "attention_mask": np.ones((1, 512), dtype=np.int64),
    }
    return service

#-----------------------------------------------------------------------------------------------------
# FastAPI test client
#-----------------------------------------------------------------------------------------------------
@pytest.fixture
def client(mock_image_service, mock_document_service):
    import os
    os.environ["API_KEYS"] = "test-key-123"

    from app.core.config import get_settings
    get_settings.cache_clear()

    from unittest.mock import MagicMock, patch

    with patch("app.features.image.service.ImageDetectionService.load_models"), \
         patch("app.features.document.service.DocumentDetectionService.load_models"), \
         patch("app.main.image_service", mock_image_service), \
         patch("app.main.document_service", mock_document_service), \
         patch("app.main.orchestrator", MagicMock()):

        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            c.headers.update({"X-API-Key": "test-key-123"})
            yield c

    get_settings.cache_clear()

@pytest.fixture
def api_key():
    return "test-key-123"