"""
Baseline Agents Implementation.

Provides:
1. Baseline 0 (Trivial): Majority-class intent + static canned reply + fixed escalation.
2. Baseline 1 (Simple): TF-IDF + Logistic Regression intent + TF-IDF cosine nearest response + single threshold escalation.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from src.intents.tfidf_classifier import TfidfIntentClassifier
from src.retrieval.tfidf_retriever import TfidfRetriever


class TrivialBaselineAgent:
    """
    Baseline 0 (Trivial):
    - Intent: Always predicts the most frequent intent (DELIVERY_SHIPPING_DELAY).
    - Reply: Static generic response.
    - Escalation: Fixed AUTO_HANDLE.
    """

    MAJORITY_INTENT = "DELIVERY_SHIPPING_DELAY"
    GENERIC_REPLY = "Thank you for reaching out to Amazon Support. Please check 'Your Orders' in your Amazon account for order updates."

    def __init__(self):
        pass

    def process_message(self, message: str) -> Dict[str, Any]:
        return {
            "message": message,
            "intent": self.MAJORITY_INTENT,
            "intent_confidence": 0.50,
            "decision": "AUTO_HANDLE",
            "escalation_reason": "Trivial baseline default auto-handle.",
            "draft_reply": self.GENERIC_REPLY,
            "retrieved_examples": []
        }


class SimpleBaselineAgent:
    """
    Baseline 1 (Simple Classical ML):
    - Intent: TF-IDF (unigrams + bigrams) + Logistic Regression.
    - Reply: Verbatim nearest historical support response via TF-IDF cosine similarity.
    - Escalation: Single confidence threshold rule (confidence < 0.50 -> ESCALATE).
    """

    def __init__(
        self,
        classifier: Optional[TfidfIntentClassifier] = None,
        retriever: Optional[TfidfRetriever] = None,
        confidence_threshold: float = 0.50
    ):
        self.classifier = classifier or TfidfIntentClassifier()
        self.retriever = retriever or TfidfRetriever()
        self.confidence_threshold = confidence_threshold

    def fit(self, train_texts: List[str], train_labels: List[str], kb_df: pd.DataFrame):
        """Fits TF-IDF classifier and indexes retrieval KB."""
        self.classifier.fit(train_texts, train_labels)
        self.retriever.build_index(kb_df)

    def process_message(self, message: str) -> Dict[str, Any]:
        # Intent prediction
        if self.classifier.is_fitted:
            intent, confidence = self.classifier.predict(message)
        else:
            intent, confidence = "OTHER_GENERAL_INQUIRY", 0.40

        # Retrieval
        if self.retriever.is_indexed:
            retrieved = self.retriever.retrieve(message, top_k=1)
            nearest_reply = retrieved[0]["historical_response"] if retrieved else "Thank you for contacting support."
            ret_score = retrieved[0]["similarity_score"] if retrieved else 0.0
        else:
            retrieved = []
            nearest_reply = "Thank you for contacting support."
            ret_score = 0.0

        # Escalation rule
        if confidence < self.confidence_threshold:
            decision = "ESCALATE"
            reason = f"Intent confidence ({confidence:.2f}) below threshold ({self.confidence_threshold:.2f})."
        else:
            decision = "AUTO_HANDLE"
            reason = f"Intent confidence ({confidence:.2f}) above threshold."

        return {
            "message": message,
            "intent": intent,
            "intent_confidence": confidence,
            "decision": decision,
            "escalation_reason": reason,
            "draft_reply": nearest_reply,
            "retrieved_examples": retrieved
        }
