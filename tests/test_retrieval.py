"""
Unit Tests for Retrieval Modules.
"""

import pytest
import pandas as pd
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.vector_retriever import VectorRetriever


@pytest.fixture
def mock_kb_df():
    return pd.DataFrame({
        "tweet_id": [101, 102, 103],
        "customer_message": [
            "Where is my package tracking update?",
            "How do I return a damaged blender?",
            "Why was my card billed twice for Prime?"
        ],
        "response_text": [
            "You can track your package directly under Your Orders.",
            "Please visit the Returns Center to print a prepaid label.",
            "Refunds for duplicate charges are processed within 3-5 business days."
        ]
    })


def test_tfidf_retriever(mock_kb_df):
    retriever = TfidfRetriever(top_k=2)
    retriever.build_index(mock_kb_df)

    results = retriever.retrieve("Where is my package?")
    assert len(results) == 2
    assert results[0]["tweet_id"] == 101
    assert "Your Orders" in results[0]["historical_response"]


def test_vector_retriever(mock_kb_df):
    retriever = VectorRetriever(top_k=1)
    retriever.build_index(mock_kb_df)

    results = retriever.retrieve("I want to return a broken product")
    assert len(results) == 1
    assert results[0]["conversation_id"] == 102
    assert retriever.get_max_similarity(results) > 0.0
