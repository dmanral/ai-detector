from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# Confidence level constants.
class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ImageSignals(BaseModel):
    """Individual signal scores that contributed to the final result."""
    fft_artifact_score:float = Field(..., ge=0.0, le=1.0, description="Unnatural frequency patterns")
    noise_pattern_score: float = Field(..., ge=0.0, le=1.0, description="Absence of natural camera noise")
    metadata_score: float = Field(..., ge=0.0, le=1.0, description="Missing or suspicious EXIF data")
    lbp_texture_score: float = Field(..., ge=0.0, le=1.0, description="Texture pattern abnormalities")
    classifier_score: float = Field(..., ge=0.0, le=1.0, description="ONNX AI image classifier")
    anatomical_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Facial geometry anomalies - null if no faces detected")

class ImageDetectionResult(BaseModel):
    """The full detection result result returned to the caller."""
    ai_probability: float = Field(..., ge=0.0, le=1.0, description="0.0 = human, 1.0 = AI")
    confidence: ConfidenceLevel
    signals: ImageSignals
    verdict: str
    face_detected: bool = Field(..., description="Whether a face was found and analyzed")