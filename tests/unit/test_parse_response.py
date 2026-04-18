"""
Tests for agent response parsing.
Both subagents use the same _parse_response pattern.
"""
import pytest
from app.agents.subagents.image_agent import ImageAgent
from app.agents.subagents.document_agent import DocumentAgent
from unittest.mock import MagicMock


@pytest.fixture
def image_agent():
    service = MagicMock()
    agent = ImageAgent(service)
    return agent


@pytest.fixture
def document_agent():
    service = MagicMock()
    agent = DocumentAgent(service)
    return agent


class TestImageAgentParseResponse:

    def test_parses_clean_format(self, image_agent):
        response = "PROBABILITY: 0.95\nVERDICT: AI generated\nREASONING: The classifier score is very high."
        result = image_agent._parse_response(response)
        assert result["probability"] == 0.95
        assert result["verdict"] == "AI generated"
        assert "classifier" in result["reasoning"].lower()

    def test_parses_lowercase_keys(self, image_agent):
        response = "probability: 0.75\nverdict: Possibly AI generated\nreasoning: Mixed signals."
        result = image_agent._parse_response(response)
        assert result["probability"] == 0.75
        assert result["verdict"] == "Possibly AI generated"

    def test_parses_bullet_point_format(self, image_agent):
        """Gemma sometimes responds with bullet points."""
        response = "* PROBABILITY: 0.88\n* VERDICT: AI generated\n* REASONING: High classifier score."
        result = image_agent._parse_response(response)
        assert result["probability"] == 0.88

    def test_clamps_probability_above_1(self, image_agent):
        response = "PROBABILITY: 1.5\nVERDICT: AI generated\nREASONING: Test."
        result = image_agent._parse_response(response)
        assert result["probability"] == 1.0

    def test_clamps_probability_below_0(self, image_agent):
        """Negative probability strings should fall back to default."""
        response = "PROBABILITY: -0.5\nVERDICT: Likely human\nREASONING: Test."
        result = image_agent._parse_response(response)
        assert 0.0 <= result["probability"] <= 1.0

    def test_fallback_on_unparseable_response(self, image_agent):
        """If Gemma returns garbage, should fall back to defaults."""
        response = "I cannot determine this from the information provided."
        result = image_agent._parse_response(response)
        assert result["probability"] == 0.5
        assert result["verdict"] == "Uncertain"
        assert result["reasoning"] == response


class TestDocumentAgentParseResponse:

    def test_parses_clean_format(self, document_agent):
        response = "PROBABILITY: 0.82\nVERDICT: AI generated\nREASONING: Uniform sentence lengths detected."
        result = document_agent._parse_response(response)
        assert result["probability"] == 0.82
        assert result["verdict"] == "AI generated"

    def test_fallback_on_missing_fields(self, document_agent):
        response = "VERDICT: Likely human"
        result = document_agent._parse_response(response)
        assert result["probability"] == 0.5  # fallback
        assert result["verdict"] == "Likely human"