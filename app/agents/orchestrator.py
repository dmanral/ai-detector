"""
Orchestrator agent.

Receives a file, determines its type, routes to the appropriate
subagent, and returns the enriched result.
"""

import logging
from typing import Optional

from app.agents.subagents.image_agent import ImageAgent
from app.agents.subagents.document_agent import DocumentAgent
from app.features.image.service import ImageDetectionService
from app.features.document.service import DocumentDetectionService
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class Orchestrator:
    """
    Routes detection requests to the appropriate subagent.
    Holds references to subagents so they're only instantiated once.
    """

    def __init__(self, image_service: ImageDetectionService, document_service: DocumentDetectionService) -> None:
        self._image_agent = ImageAgent(image_service)
        self._document_agent = DocumentAgent(document_service)
        logger.info("Orchestrator initialised with image and document subagents.")
    
    async def run(self, contents: bytes, content_type: str) -> dict:
        """
        Route the file to the right subagent and return the result.
        """
        logger.info("orchestrator_run content_type=%s size=%d", content_type, len(contents))

        if self._is_image(content_type):
            return await self._image_agent.analyze(contents, content_type)
        elif self._is_document(content_type):
            return await self._document_agent.analyze(contents, content_type)
        else:
            raise ValueError(f"Unsupported conent type: {content_type}")

    def _is_image(self, content_type: str) -> bool:
        return content_type in settings.allowed_image_types
    
    def _is_document(self, content_type: str) -> bool:
        return content_type in settings.allowed_document_types