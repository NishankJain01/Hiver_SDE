"""
TF-IDF Baseline Conversation Retriever.

Indexes historical customer queries using TF-IDF and returns
the most similar historical support responses via cosine similarity.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """
    TF-IDF based Historical Response Retriever.
    """

    def __init__(self, top_k: int = 3, max_features: int = 15000):
        self.top_k = top_k
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.kb_df: Optional[pd.DataFrame] = None
        self.doc_vectors = None
        self.is_indexed = False

    def build_index(self, kb_df: pd.DataFrame, text_col: str = "customer_message"):
        """Indexes historical customer support dialogues."""
        self.kb_df = kb_df.reset_index(drop=True).copy()
        texts = self.kb_df[text_col].fillna("").astype(str).tolist()
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self.is_indexed = True
        return self

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most similar historical conversations.

        Returns:
            List of dictionaries with:
            - tweet_id
            - customer_message
            - response_text
            - similarity_score
        """
        if not self.is_indexed or self.kb_df is None:
            raise RuntimeError("Knowledge base must be indexed before calling retrieve.")

        k = top_k or self.top_k
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors)[0]

        top_indices = np.argsort(scores)[::-1][:k]
        results = []

        for idx in top_indices:
            score = float(scores[idx])
            row = self.kb_df.iloc[idx]
            results.append({
                "tweet_id": int(row.get("tweet_id", idx)),
                "historical_customer_message": str(row.get("customer_message", "")),
                "historical_response": str(row.get("response_text", "")),
                "similarity_score": round(score, 3),
                "retrieval_method": "tfidf_cosine"
            })

        return results
