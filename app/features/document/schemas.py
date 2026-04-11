from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class DocumentSignals(BaseModel):
    perplexity_score: float = Field(..., ge=0, le=1.0, description="How predictable the text is - low perplexity = AI")
    burstiness_score: float = Field(..., ge=0, le=1.0, description="Sentence length variance - uniform = AI")
    stylometric_score: float = Field(..., ge=0, le=1.0, description="Vocab richness, function word patterns, etc.")
    repetition_score: float = Field(..., ge=0, le=1.0, description="N-gram repetition patterns - high repetition = AI")
    classifier_score: float = Field(..., ge=0, le=1.0, description="RoBERTa AI classifier confidence")

class DocumentDetectionResult(BaseModel):
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: ConfidenceLevel
    signals: DocumentSignals
    verdict: str
    word_count: int = Field(..., description="Number of words analyzed")
    char_count: int = Field(..., description="Number of characters analyzed")
    truncated: bool = Field(..., description="Whether text was truncated due to length")
    reliability_note: str = "Text detection accuracy is lower for modern AI models."
