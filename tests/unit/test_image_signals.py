"""
Tests for individual image detection signals.
These test the math/logic of each signal function directly.
"""
import io
import pytest
import numpy as np
from PIL import Image
from app.features.image.service import ImageDetectionService


@pytest.fixture
def service():
    """Service instance with no models loaded — signals don't need them."""
    return ImageDetectionService()


@pytest.fixture
def solid_image():
    """A perfectly uniform solid colour image — no noise, no texture."""
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def noisy_image():
    """An image with random noise — simulates real camera sensor noise."""
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


class TestFFTArtifactScore:

    def test_returns_float_between_0_and_1(self, service, solid_image):
        score = service._fft_artifact_score(solid_image)
        assert 0.0 <= score <= 1.0

    def test_noisy_image_returns_valid_score(self, service, noisy_image):
        score = service._fft_artifact_score(noisy_image)
        assert 0.0 <= score <= 1.0

    def test_does_not_crash_on_small_image(self, service):
        tiny = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8), "RGB")
        score = service._fft_artifact_score(tiny)
        assert 0.0 <= score <= 1.0


class TestNoisePatternScore:

    def test_returns_float_between_0_and_1(self, service, solid_image):
        score = service._noise_pattern_score(solid_image)
        assert 0.0 <= score <= 1.0

    def test_solid_image_scores_high(self, service, solid_image):
        """A solid image has no noise — should score suspiciously clean."""
        score = service._noise_pattern_score(solid_image)
        assert score >= 0.8

    def test_noisy_image_scores_lower(self, service, noisy_image):
        """A noisy image looks more like a real camera photo."""
        score = service._noise_pattern_score(noisy_image)
        assert score < 0.8


class TestMetadataScore:

    def test_no_exif_returns_moderate_score(self, service):
        score = service._metadata_score({})
        assert score == 0.65

    def test_ai_tool_in_metadata_returns_high_score(self, service):
        exif = {271: "DALL-E generated image"}
        score = service._metadata_score(exif)
        assert score == 0.99

    def test_full_camera_metadata_returns_low_score(self, service):
        exif = {
            "Make": "Canon",
            "Model": "EOS R5",
            "LensModel": "RF 50mm",
            "DateTime": "2024:01:01 12:00:00",
            "GPSInfo": {"lat": 37.7}
        }
        score = service._metadata_score(exif)
        assert score < 0.65

    def test_midjourney_in_metadata(self, service):
        exif = {270: "Created with Midjourney v6"}
        score = service._metadata_score(exif)
        assert score == 0.99


class TestLBPTextureScore:

    def test_returns_float_between_0_and_1(self, service, solid_image):
        score = service._lbp_texture_score(solid_image)
        assert 0.0 <= score <= 1.0

    def test_solid_image_scores_low(self, service, solid_image):
        """Solid image has perfectly uniform LBP — deviation is 0, scores 0.0."""
        score = service._lbp_texture_score(solid_image)
        assert score == 0.0

    def test_noisy_image_scores_lower(self, service, noisy_image):
        """Random noise produces varied LBP histogram — more human-like."""
        score = service._lbp_texture_score(noisy_image)
        assert score <= 0.5