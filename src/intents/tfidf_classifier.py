"""
TF-IDF + Logistic Regression Intent Classifier (Baseline 1).

Trains an explainable linear model on n-gram TF-IDF representations.
Outputs class predictions and calibrated probability confidence.
"""

import os
import joblib
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfIntentClassifier:
    """
    Classical Machine Learning baseline using TF-IDF and Logistic Regression.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        max_iter: int = 1000,
        random_state: int = 42
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b"
            )),
            ("clf", LogisticRegression(
                max_iter=self.max_iter,
                class_weight="balanced",
                C=1.0,
                random_state=self.random_state
            ))
        ])
        self.is_fitted = False
        self.classes_: Optional[np.ndarray] = None

    def fit(self, texts: List[str], labels: List[str]):
        """Fits the vectorizer and logistic regression classifier."""
        cleaned_texts = [t if isinstance(t, str) else "" for t in texts]
        self.pipeline.fit(cleaned_texts, labels)
        self.is_fitted = True
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predicts intent and confidence score for a single customer message.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict.")

        clean_text = text if isinstance(text, str) else ""
        probas = self.pipeline.predict_proba([clean_text])[0]
        best_idx = np.argmax(probas)
        best_class = self.classes_[best_idx]
        confidence = float(probas[best_idx])

        return best_class, round(confidence, 3)

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predicts intents and confidences for a batch of messages."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict_batch.")

        cleaned_texts = [t if isinstance(t, str) else "" for t in texts]
        probas = self.pipeline.predict_proba(cleaned_texts)
        results = []
        for prob_vec in probas:
            best_idx = np.argmax(prob_vec)
            best_class = self.classes_[best_idx]
            confidence = float(prob_vec[best_idx])
            results.append((best_class, round(confidence, 3)))
        return results

    def save(self, filepath: str):
        """Serializes the trained pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)

    def load(self, filepath: str):
        """Loads a saved pipeline from disk."""
        self.pipeline = joblib.load(filepath)
        self.is_fitted = True
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self
