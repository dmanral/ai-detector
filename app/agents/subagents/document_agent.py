"""
Document detectin subagent.

Calls the existing document detection pipeline to get signal scores,
then sends both scores and the actual text to Gemma for reasoning.
Returns an enriched result with the original signals plus the LLM reasoning.
"""

import logging

from app.agents.base import BaseAgent
from app.features.document.service import DocumentDetectionService
from app.features.document.schemas import DocumentDetectionResult

logger = logging.getLogger(__name__)

# Max number of chars of text to send to Gemma
# sending too much slows inference and hits conext limits.
MAX_TEXT_FOR_LLM = 3000

class DocumentAgent(BaseAgent):

    def __init__(self, document_service: DocumentDetectionService) -> None:
        super().__init__()
        self._document_service = document_service
    
    async def analyze(self, contents: bytes, content_type: str) -> dict:
        """
        1. Run the existing signal pipeline.
        2. Extract the text for Gemma.
        3. Send signals + text to Gemma for reasoning.
        4. Return enriched results.
        """

        self._log_analysis_start("document", content_type=content_type)

        # Step 1 - run existing detection pipeline
        signal_result: DocumentDetectionResult = await self._document_service.analyze(contents, content_type)

        # Step 2 - extract text to send to Gemma
        # We re-extracting so we have the raw text for the prompt.
        text = self._document_service._extract_text(contents, content_type)
        text_for_llm = text[:MAX_TEXT_FOR_LLM]
        if len(text) > MAX_TEXT_FOR_LLM:
            text_for_llm += "\n\n[Text truncated for analysis]"

        # Step 3 - build prompt with signal scores and text.
        template = self.load_prompt("document_prompt.txt")
        prompt = template.format(
            perplexity_score=signal_result.signals.perplexity_score,
            burstiness_score=signal_result.signals.burstiness_score,
            stylometric_score=signal_result.signals.stylometric_score,
            repetition_score=signal_result.signals.repetition_score,
            classifier_score=signal_result.signals.classifier_score,
            word_count=signal_result.word_count,
            text=text_for_llm,
        )

        # Step 4 - send to Gemma.
        raw_response = await self.llm.complete(
            prompt=prompt,
            system="You are an expert AI text detection analyst. Be precise and reference specific parts of the text in your reasoning."
        )

        # Step 5 - parse Gemma's response.
        parsed = self._parse_response(raw_response)

        self._log_analysis_complete("document", parsed["verdict"], parsed["probability"])

        # Step 6 - return the enriched result
        return {
            "ai_probability": parsed["probability"],
            "confidence": signal_result.confidence,
            "signals": signal_result.signals.model_dump(),
            "verdict": parsed["verdict"],
            "word_count": signal_result.word_count,
            "char_count": signal_result.char_count,
            "truncated": signal_result.truncated,
            "agent_reasoning": parsed["reasoning"],
            "pipeline_probability": signal_result.ai_probability,
        }
    
    def _parse_response(self, response: str) -> dict:
        """
        Parse Gemma's structured response.
        Expected format:
            PROBABILITY: 0.87
            VERDICT: AI generated
            REASONING: ...
        """

        lines = response.strip().split("\n")
        result = {
            "probability": 0.5,
            "verdict": "Uncertain",
            "reasoning": response
        }

        for line in lines:
            line = line.strip()
            if line.startswith("PROBABILITY:"):
                try:
                    val = float(line.split(":", 1)[1].strip())
                    result["probability"] = max(0.0, min(1.0, val))
                except ValueError:
                    pass
            elif line.startswith("VERDICT:"):
                result["verdict"] = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
        
        return result
