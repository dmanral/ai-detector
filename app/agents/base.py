"""
Base agent class, all the subagents will inherit from this class.

Provides:
    - Access to the LLM client.
    - Standard interface every subagent must implement (e.g. `run` method).
    - Shared prompt loading utilities.
    - Consistent logging setup for all agents.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.agents.tools.llm_client import LLMClient

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

class BaseAgent(ABC):
    """
    Every subagent inherits from this and implements analyze and run methods.
    """

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def analyze(self, *args: Any, **kwargs: Any) -> str:
        """
        Run the agent's analysis and return a result in a dict.
        Every subagent must implement this method.
        """
        ...
    
    def load_prompt(self, filename: str) -> str:
        """
        Load a prompt template from the prompts directory.

        Use:
            template = self.load_prompt("image_prompt.txt")
            prompt = template.format(signal_1=0.99, signal_2=0.45)
        """

        path = PROMPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text().strip()
    
    def _log_analysis_start(self, media_type: str, **kwargs: Any) -> None:
        self.logger.info(
            "agent_analysis_start type=%s %s",
            media_type,
            " ".join(f"{k}={v}" for k, v in kwargs.items()),
        )
    
    def _log_analysis_complete(self, media_type: str, verdict: str, probability: float) -> None:
        self.logger.info(
            "agent_analysis_complete type=%s verdict=%s probability=%.3f",
            media_type,
            verdict,
            probability,
        )