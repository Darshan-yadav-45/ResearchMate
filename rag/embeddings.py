"""Vector embedding and similarity computation for academic RAG."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class TextEmbeddingEngine:
    """Lightweight, resilient text embedding engine based on TF-IDF + n-gram vectorization.
    Computes exact cosine similarity scores without external cloud dependencies or latency.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            max_features=10000,
        )
        self._fitted = False

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        """Fit vectorizer on document chunks and transform to vector space."""
        if not corpus:
            return np.empty((0, 0))
        vectors = self.vectorizer.fit_transform(corpus)
        self._fitted = True
        return vectors

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to the fitted vector space."""
        if not self._fitted:
            return self.fit_transform(texts)
        return self.vectorizer.transform(texts)

    def compute_similarity(
        self, query: str, document_vectors: np.ndarray
    ) -> np.ndarray:
        """Computes cosine similarity between query and all document vectors.

        Args:
            query: The user query string.
            document_vectors: Matrix of document chunk vectors.

        Returns:
            1D numpy array of similarity scores [0.0 to 1.0].
        """
        if not self._fitted or document_vectors.shape[0] == 0:
            return np.array([])

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, document_vectors).flatten()
        return scores
