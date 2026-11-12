"""AI package for IBM Granite and watsonx.ai integration."""
from ai.watsonx_client import (
    WatsonxClient,
    WatsonxConfigurationError,
    WatsonxAPIError,
)
from ai.prompts import (
    build_question_analysis_prompt,
    build_paper_summary_prompt,
    build_literature_review_prompt,
    build_research_gap_prompt,
    build_report_draft_prompt,
    build_rag_qa_prompt,
)

__all__ = [
    "WatsonxClient",
    "WatsonxConfigurationError",
    "WatsonxAPIError",
    "build_question_analysis_prompt",
    "build_paper_summary_prompt",
    "build_literature_review_prompt",
    "build_research_gap_prompt",
    "build_report_draft_prompt",
    "build_rag_qa_prompt",
]
