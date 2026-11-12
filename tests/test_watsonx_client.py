"""Unit tests for IBM watsonx.ai client and prompt engineering."""

import pytest
from ai.watsonx_client import WatsonxClient, WatsonxConfigurationError
from ai.prompts import (
    build_question_analysis_prompt,
    build_paper_summary_prompt,
    build_literature_review_prompt,
    build_research_gap_prompt,
    build_report_draft_prompt,
    build_rag_qa_prompt,
)


def test_watsonx_client_unconfigured():
    client = WatsonxClient(apikey="", project_id="")
    assert not client.is_configured()

    with pytest.raises(WatsonxConfigurationError):
        client.generate(prompt="Hello Granite")


def test_question_analysis_prompt_structure():
    prompt = build_question_analysis_prompt(topic="Edge AI Quantization", background_context="IoT devices")
    assert "Edge AI Quantization" in prompt
    assert "IoT devices" in prompt
    assert "Research Objectives" in prompt
    assert "Recommended Academic Search Queries" in prompt


def test_paper_summary_prompt_structure():
    prompt = build_paper_summary_prompt(
        paper_title="Test Title",
        authors="A. Author",
        paper_text="This paper introduces a novel pruning technique.",
    )
    assert "Test Title" in prompt
    assert "1. Research Objectives" in prompt
    assert "2. Abstract Summary" in prompt
    assert "3. Methodology" in prompt
    assert "4. Key Findings" in prompt
    assert "5. Limitations" in prompt
    assert "6. Future Research Directions" in prompt


def test_report_draft_prompt_integrity_markers():
    prompt = build_report_draft_prompt(topic="Quantum Machine Learning", lit_review="", papers_summary="")
    assert "Quantum Machine Learning" in prompt
    assert "[ORIGINAL RESEARCH REQUIRED:" in prompt
    assert "Proposed Methodology" in prompt
    assert "Expected Results vs. Actual Results" in prompt
