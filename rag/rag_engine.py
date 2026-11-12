"""Retrieval-Augmented Generation (RAG) Orchestration Engine for ResearchMate."""

import logging
from typing import List, Dict, Any, Optional

from rag.vector_store import VectorStore, global_vector_store
from ai.watsonx_client import WatsonxClient, WatsonxConfigurationError, WatsonxAPIError
from ai.prompts import build_rag_qa_prompt

logger = logging.getLogger(__name__)


class RAGEngine:
    """Coordinates document retrieval and grounded IBM Granite question answering."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or global_vector_store
        self.watsonx_client = WatsonxClient()

    def query(
        self,
        question: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Performs RAG: retrieves relevant chunks and queries IBM Granite.

        Args:
            question: Natural-language research query.
            top_k: Number of chunks to retrieve.
            paper_ids: Optional list of papers to restrict search to.

        Returns:
            Dict containing answer, sources, retrieved chunks, and model metadata.
        """
        # Step 1: Vector search
        retrieved_chunks = self.vector_store.search(
            query=question, top_k=top_k, paper_ids=paper_ids
        )

        if not retrieved_chunks:
            return {
                "answer": "No relevant document chunks found matching your query. Please upload research papers or expand your search question.",
                "retrieved_chunks": [],
                "sources": [],
                "model_used": None,
                "is_ai_generated": False,
            }

        # Format sources
        sources = []
        for c in retrieved_chunks:
            source_entry = {
                "paper_title": c.get("paper_title", "Document"),
                "page_number": c.get("page_number", 1),
                "similarity_score": c.get("similarity_score", 0.0),
                "section": c.get("section_hint", "Body"),
                "preview": c.get("content", "")[:200] + "...",
            }
            if source_entry not in sources:
                sources.append(source_entry)

        # Step 2: Check IBM Granite availability
        if not self.watsonx_client.is_configured():
            return {
                "answer": (
                    "**IBM Granite Configuration Required**\n\n"
                    "Evidence was successfully retrieved from your uploaded papers (see Retrieved Sources below). "
                    "However, to generate an AI synthesis, you must configure your IBM watsonx.ai credentials (`WATSONX_APIKEY` and `WATSONX_PROJECT_ID`) in the **Settings** page or `.env` file."
                ),
                "retrieved_chunks": retrieved_chunks,
                "sources": sources,
                "model_used": None,
                "is_ai_generated": False,
                "setup_required": True,
            }

        # Step 3: Send to IBM Granite
        prompt = build_rag_qa_prompt(query=question, retrieved_chunks=retrieved_chunks)

        try:
            granite_res = self.watsonx_client.generate(
                prompt=prompt,
                max_new_tokens=1000,
                temperature=0.2,  # Low temperature for factual grounding
            )
            answer_text = granite_res.get("generated_text", "")
            return {
                "answer": answer_text,
                "retrieved_chunks": retrieved_chunks,
                "sources": sources,
                "model_used": granite_res.get("model_id"),
                "is_ai_generated": True,
                "tokens": {
                    "input": granite_res.get("input_token_count", 0),
                    "output": granite_res.get("generated_token_count", 0),
                },
            }
        except (WatsonxConfigurationError, WatsonxAPIError) as e:
            logger.error(f"Error querying IBM Granite in RAG engine: {e}")
            return {
                "answer": f"**IBM Granite Error**: {str(e)}\n\nRetrieved document evidence is still accessible below.",
                "retrieved_chunks": retrieved_chunks,
                "sources": sources,
                "model_used": None,
                "is_ai_generated": False,
                "error": str(e),
            }
