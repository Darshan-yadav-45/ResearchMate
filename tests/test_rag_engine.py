"""Unit tests for RAG Chunking, Embeddings, and Vector Search."""

import pytest
from rag.chunker import TextChunker
from rag.embeddings import TextEmbeddingEngine
from rag.vector_store import VectorStore
from rag.rag_engine import RAGEngine


def test_text_chunking():
    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    pages = [
        {
            "page_number": 1,
            "text": "Abstract: Deep learning models have grown rapidly in size. This paper investigates memory efficiency.",
        },
        {
            "page_number": 2,
            "text": "Methodology: We implement an 8-bit quantization framework. Weights are clustered using k-means.",
        },
    ]

    chunks = chunker.chunk_page_texts(pages, paper_id=101)
    assert len(chunks) >= 2
    assert chunks[0]["paper_id"] == 101
    assert chunks[0]["page_number"] == 1
    assert "Abstract" in chunks[0]["section_hint"]


def test_embedding_and_similarity():
    engine = TextEmbeddingEngine()
    corpus = [
        "Quantization and pruning reduce neural network memory consumption.",
        "Computer vision models classify images using convolutional layers.",
        "Natural language processing translates sentences between languages.",
    ]

    vectors = engine.fit_transform(corpus)
    assert vectors.shape[0] == 3

    # Query related to quantization
    scores = engine.compute_similarity("neural network memory quantization", vectors)
    assert len(scores) == 3
    # First chunk should have the highest similarity score
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_vector_store_retrieval():
    store = VectorStore()
    test_chunks = [
        {
            "paper_id": 1,
            "paper_title": "Paper A",
            "chunk_index": 0,
            "content": "Reinforcement learning optimizes decision-making policies using reward signals.",
            "page_number": 1,
            "section_hint": "Introduction",
        },
        {
            "paper_id": 2,
            "paper_title": "Paper B",
            "chunk_index": 0,
            "content": "Convolutional networks excel at spatial feature extraction in image classification.",
            "page_number": 3,
            "section_hint": "Methodology",
        },
    ]

    store.set_chunks(test_chunks)
    assert len(store.chunks) == 2

    results = store.search("reward signals policy optimization", top_k=1)
    assert len(results) == 1
    assert results[0]["paper_title"] == "Paper A"
    assert results[0]["similarity_score"] > 0.1


def test_rag_engine_unconfigured_behavior():
    store = VectorStore()
    store.set_chunks([
        {
            "paper_id": 1,
            "paper_title": "Sample Paper",
            "chunk_index": 0,
            "content": "Transformers utilize self-attention mechanisms to process tokens in parallel.",
            "page_number": 2,
            "section_hint": "Architecture",
        }
    ])
    engine = RAGEngine(vector_store=store)
    res = engine.query("How do transformers process tokens?")
    assert len(res["retrieved_chunks"]) == 1
    assert res["retrieved_chunks"][0]["paper_title"] == "Sample Paper"
    assert "retrieved" in res["answer"].lower() or "granite" in res["answer"].lower()
