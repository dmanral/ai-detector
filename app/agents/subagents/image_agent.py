"""
Image detection subagent.

Calls the existing image detection pipeline to get signal scores.
It then sends both the scores and the actual image to Gemma for reasoning.
Returns an enriched result with the original signals plus LLM reasoing.
"""

import logging
from typing import Optional

from app.agents.base import BaseAgent
from app.features.image.service import ImageDetectionService
from app.features.image.schemas import ImageDetectionResult
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class ImageAgent(BaseAgent):

    def __init__(self, image_service: ImageDetectionService) -> None:
        super().__init__()

        # Reause the already-loaded service instance from main.py
        # so we don't reload models on every request.
        self._image_service = image_service
    
    async def analyze(self, contents: bytes, content_type: str) -> dict:
        """
        1. Run the existing signal pipeline to get scores.
        2. Send signals + image to Gemma for reasoning.
        3. Parse Gemma's reasonings.
        4. Return Gemma's enriched result, which includes the original signals plus LLM reasoning.
        """
        self._log_analysis_start("image", content_type=content_type)

        # Step 1: run the existing detection pipeline.
        signal_result: ImageDetectionResult = await self._image_service.analyze(contents)

        # Step 2: Build the prompt with signal scores.
        template = self.load_prompt("image_prompt.txt")
        prompt = template.format(
            fft_artifact_score=signal_result.signals.fft_artifact_score,
            noise_pattern_score=signal_result.signals.noise_pattern_score,
            metadata_score=signal_result.signals.metadata_score,
            lbp_texture_score=signal_result.signals.lbp_texture_score,
            classifier_score=signal_result.signals.classifier_score,
            anatomical_score=signal_result.signals.anatomical_score or "n/a — no face detected",
            face_detected=signal_result.face_detected,
        )

        # Temporary debug line
        logger.info("PROMPT PREVIEW: %s", prompt[:200])

        # Step 3: Send the Gemma with the actual image.
        raw_response = await self.llm.complete(
            prompt=prompt,
            system="You are an expert AI image detection analyst. Be precise and analytical.",
            image_bytes=contents if settings.llm_provider == "google" else None,
        )

        # Step 4: Parse Gemma's response.
        parsed = self._parse_response(raw_response)

        self._log_analysis_complete("image", parsed["verdict"], parsed["probability"])

        # Step 5 - Return enriched result combining signals and reasoning.
        return{
            "ai_probability": parsed["probability"],
            "confidence": signal_result.confidence,
            "signals": signal_result.signals.model_dump(),
            "verdict": parsed["verdict"],
            "face_detected": signal_result.face_detected,
            "agent_reasoning": parsed["reasoning"],
            "pipeline_probability": signal_result.ai_probability,
        }
    
    def _parse_response(self, response: str) -> dict:
        """
        Parse Gemma's structured response.
        Expected format:
            Probability: 0.95,
            Verdict: AI generated,
            Reasoning: ...
        """
        lines = response.strip().split("\n")
        result = {
            "probability": 0.5,
            "verdict": "Uncertain",
            "reasoning": response,  # Fallback to full response if parsing fails.
        }

        for line in lines:
            line = line.strip().lstrip("*-• ").strip()

            # Handle PROBABILITY
            if line.lower().startswith("probability:"):
                try:
                    val_srt = line.split(":", 1)[1].strip()
                    # Extract first number found.
                    import re
                    numbers = re.findall(r"0?\.\d+|\d+\.?\d*", val_str)
                    if numbers:
                        val = float(numbers[0])
                        result["probability"] = max(0.0, min(1.0, val))
                except (ValueError, IndexError):
                    pass
            
            # Handle VERDICT
            elif line.lower().startswith("verdict:"):
                result["verdict"] = line.split(":", 1)[1].strip()
            
            # Handle REASONING
            elif line.lower().startswith("reasoning:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
        
        return result