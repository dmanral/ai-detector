from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        emv_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "AI Detector API"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development" # development | staging | production

    # --- Auth ---
    api_key_header_name: str = "X-API-Key"
    api_keys: str = "dev-key-1"      # comma-separated, override in .env

    # --- Model paths ---
    efficientnet_model_dir: str = "./app/models/classifier"
    dlib_landmark_model: str = "./app/models/shape_predictor_68_face_landmarks.dat"

    # --- Detection thresholds ---
    ai_verdict_threshold: float = 0.75
    possible_ai_verdict_threshold: float = 0.45

    # --- File size limits ---
    max_image_size_mb: int = 40
    max_video_size_mb: int = 700
    max_document_size_mb: int = 20

    # --- Allowed MIME types ---
    allowed_image_types: list[str] = ["image/jpeg", "image/png", "image/webp"]
    allowed_video_types: list[str] = ["video/mp4", "video/quicktime", "video/webm"]
    allowed_document_types: list[str] = [
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]

    # --- Async jobs ---
    job_result_ttl_seconds: int = 3600  # how long to keep results (1 hour)
    redis_url: str = "redis://localhost:6379/0" # for async job queue later

    # --- Text classifier ---
    text_classifier_model: str = "./app/models/text_classifier/model.onnx"
    text_classifier_tokenizer_dir: str = "./app/models/text_classifier/tokenizer"

    # --- Computed helpers (no env var needed) ---
    @property
    def api_keys_set(self) -> set[str]:
        return set(k.strip() for k in self.api_keys.split(",") if k.strip())
    
    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024
    
    @property
    def max_video_size_bytes(self) -> int:
        return self.max_video_size_mb * 1024 * 1024

    @property
    def max_document_size_bytes(self) -> int:
        return self.max_document_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()
"""

---

## The @property fields - why they exist

The three _bytes properties and api_keys_set are not settings you configure - they're **derived values** computed from your real settings.
So when your upload validator needs to check a file size, it calls settings.max_image_size_bytes and gets a number in bytes directly, without doing mb * 1024 * 1024 inline all over the codebase. 
Same idea for api_keys_set - you store one env var as a comma-separated string but use it everywhere as a proper Python set.

---

"""
