"""Vector store management for ResearchMate."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from rag.embeddings import TextEmbeddingEngine

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document chunk storage and vector similarity retrieval."""

    def __init__(self):
        self.embedding_engine = TextEmbeddingEngine()
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> int:
        """Add new chunks to the vector store and re-index.

        Args:
            new_chunks: List of chunk dicts (must have 'content', 'paper_id', etc.)

        Returns:
            Total number of chunks in the store.
        """
        if not new_chunks:
            return len(self.chunks)

        self.chunks.extend(new_chunks)
        self._reindex()
        return len(self.chunks)

    def set_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Replace all chunks in store with given chunks and index."""
        self.chunks = list(chunks)
        self._reindex()
        return len(self.chunks)

    def _reindex(self):
        """Fit and transform vectors across all current chunks."""
        if not self.chunks:
            self.vectors = None
            return

        corpus = [c.get("content", "") for c in self.chunks]
        self.vectors = self.embedding_engine.fit_transform(corpus)
        logger.info(f"Vector store indexed {len(self.chunks)} chunks.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
        min_score: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Search for chunks most similar to the query.

        Args:
            query: User's search query or research question.
            top_k: Maximum number of chunks to return.
            paper_ids: Optional list of paper IDs to restrict search to.
            min_score: Minimum similarity score cutoff.

        Returns:
            List of chunk dicts with added 'similarity_score'.
        """
        if not self.chunks or self.vectors is None:
            return []

        # If filtering by paper_ids
        if paper_ids:
            paper_ids_set = set(paper_ids)
            filtered_indices = [
                i for i, c in enumerate(self.chunks) if c.get("paper_id") in paper_ids_set
            ]
            if not filtered_indices:
                return []
            sub_vectors = self.vectors[filtered_indices]
            sub_chunks = [self.chunks[i] for i in filtered_indices]
        else:
            sub_vectors = self.vectors
            sub_chunks = self.chunks

        scores = self.embedding_engine.compute_similarity(query, sub_vectors)
        if len(scores) == 0:
            return []

        # Get top-k indices sorted descending
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score >= min_score or len(results) == 0:
                chunk_copy = dict(sub_chunks[idx])
                chunk_copy["similarity_score"] = round(score, 4)
                results.append(chunk_copy)

        return results

    def clear(self):
        """Clears all stored chunks and vectors."""
        self.chunks.clear()
        self.vectors = None


# Global singleton vector store instance
global_vector_store = VectorStore()

