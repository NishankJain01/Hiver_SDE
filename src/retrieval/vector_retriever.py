"""
Semantic Vector Knowledge Base Retriever (Main System).

Indexes historical customer-support conversations and provides
dense semantic retrieval with similarity scoring, evidence formatting,
and grounding confidence bounds.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorRetriever:
    """
    Dense Semantic Retriever for Grounded Response Generation.
    """

    def __init__(
        self,
        top_k: int = 3,
        similarity_threshold: float = 0.55,
        min_evidence_score: float = 0.40
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.min_evidence_score = min_evidence_score

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=25000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.kb_df: Optional[pd.DataFrame] = None
        self.doc_vectors = None
        self.is_indexed = False

    def build_index(
        self,
        kb_df: pd.DataFrame,
        customer_col: str = "customer_message",
        response_col: str = "response_text"
    ):
        """Indexes historical customer interactions."""
        self.kb_df = kb_df.reset_index(drop=True).copy()
        texts = self.kb_df[customer_col].fillna("").astype(str).tolist()
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self.is_indexed = True
        return self

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_intent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k historical support dialogues for evidence grounding.

        Returns:
            List of evidence dictionaries with similarity scores and metadata.
        """
        if not self.is_indexed or self.kb_df is None:
            raise RuntimeError("Knowledge Base must be indexed before calling retrieve.")

        k = top_k or self.top_k
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors)[0]

        top_indices = np.argsort(scores)[::-1][:k]
        evidence_list = []

        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            row = self.kb_df.iloc[idx]

            evidence = {
                "rank": rank + 1,
                "conversation_id": int(row.get("tweet_id", idx)),
                "similarity_score": round(score, 3),
                "historical_customer_message": str(row.get("customer_message", "")),
                "historical_support_response": str(row.get("response_text", "")),
                "is_sufficient_evidence": bool(score >= self.similarity_threshold)
            }
            evidence_list.append(evidence)

        return evidence_list

    def format_evidence_for_prompt(self, evidence_list: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved evidence list into structured text for LLM injection.
        """
        if not evidence_list:
            return "No historical support evidence available."

        formatted_chunks = []
        for ev in evidence_list:
            chunk = (
                f"[Evidence #{ev['rank']} | Similarity: {ev['similarity_score']:.2f}]\n"
                f"Customer: {ev['historical_customer_message']}\n"
                f"Amazon Support: {ev['historical_support_response']}"
            )
            formatted_chunks.append(chunk)

        return "\n\n".join(formatted_chunks)

    def get_max_similarity(self, evidence_list: List[Dict[str, Any]]) -> float:
        """Returns highest similarity score among retrieved items."""
        if not evidence_list:
            return 0.0
        return max(ev["similarity_score"] for ev in evidence_list)
