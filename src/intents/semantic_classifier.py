"""
Semantic and Hybrid Intent Classifier (Main System).

Leverages semantic exemplar matching, centroid cosine similarity,
and optional LLM zero-shot classification to achieve superior Macro F1
across nuanced intent boundaries.
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticIntentClassifier:
    """
    Semantic Intent Classifier leveraging curated intent definitions,
    exemplars, and hybrid semantic projection.
    """

    def __init__(self, config_path: str = "config/intents.yaml", llm_client: Optional[Any] = None):
        self.config_path = config_path
        self.llm_client = llm_client
        self.intents_data = self._load_config()
        self.intent_ids = [item["id"] for item in self.intents_data]
        self.intent_descriptions = {item["id"]: item["description"] for item in self.intents_data}
        self.intent_examples = {item["id"]: item["positive_examples"] for item in self.intents_data}

        # Build exemplar semantic vectorizer
        self._build_semantic_index()

    def _load_config(self) -> List[Dict[str, Any]]:
        """Loads intent definitions from YAML config."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Intents configuration not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("intents", [])

    def _build_semantic_index(self):
        """Constructs semantic profile vectors for each intent class."""
        self.corpus_docs = []
        self.corpus_labels = []

        for item in self.intents_data:
            intent_id = item["id"]
            desc = item["description"]
            boundary = item.get("boundary_notes", "")
            examples = item.get("positive_examples", [])

            # Combine description and examples for rich class centroid
            full_text = f"{desc} {boundary} {' '.join(examples)}"
            self.corpus_docs.append(full_text)
            self.corpus_labels.append(intent_id)

            for ex in examples:
                self.corpus_docs.append(ex)
                self.corpus_labels.append(intent_id)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.corpus_docs)

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predicts intent class and confidence score.

        If LLM client is available and active, can query LLM with fallback to semantic vectors.
        """
        if not text or not isinstance(text, str):
            return "OTHER_GENERAL_INQUIRY", 0.40

        # If LLM client is configured and not mock, attempt LLM structured prediction
        if self.llm_client is not None and getattr(self.llm_client, "is_active", False):
            try:
                llm_res = self.llm_client.classify_intent(text)
                if llm_res and "intent" in llm_res and llm_res["intent"] in self.intent_ids:
                    return llm_res["intent"], float(llm_res.get("confidence", 0.85))
            except Exception:
                pass  # Fallback to dense exemplar similarity

        # Semantic exemplar cosine similarity
        query_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]

        # Aggregate similarity scores per intent
        intent_scores = {intent_id: [] for intent_id in self.intent_ids}
        for score, label in zip(similarities, self.corpus_labels):
            intent_scores[label].append(score)

        # Compute max + top mean score per intent with domain keyword boost
        text_lower = text.lower()
        aggregated_scores = {}
        for intent_id, scores in intent_scores.items():
            sorted_scores = sorted(scores, reverse=True)
            top_score = sorted_scores[0] if sorted_scores else 0.0
            avg_top = np.mean(sorted_scores[:2]) if len(sorted_scores) >= 2 else top_score
            base_score = 0.6 * top_score + 0.4 * avg_top

            # Keyword boosting
            boost = 0.0
            if intent_id == "ACCOUNT_ACCESS_SECURITY" and any(k in text_lower for k in ["password", "otp", "2fa", "login", "hacked", "lockout", "compromised", "authenticator"]):
                boost = 0.25
            elif intent_id == "REFUND_BILLING_INQUIRY" and any(k in text_lower for k in ["refund", "money back", "charged", "billing", "fee", "debit"]):
                boost = 0.20
            elif intent_id == "RETURN_EXCHANGE_REQUEST" and any(k in text_lower for k in ["return", "exchange", "pickup", "label", "replacement", "drop off"]):
                boost = 0.20
            elif intent_id == "DELIVERY_SHIPPING_DELAY" and any(k in text_lower for k in ["delivery", "delivered", "shipping", "courier", "package", "tracking", "ontrac", "usps"]):
                boost = 0.20
            elif intent_id == "DAMAGED_DEFECTIVE_PRODUCT" and any(k in text_lower for k in ["damaged", "broken", "defective", "shattered", "leaked", "scratch", "swollen", "fire"]):
                boost = 0.25
            elif intent_id == "ORDER_CANCELLATION_MODIFICATION" and any(k in text_lower for k in ["cancel", "cancellation", "change address", "modify"]):
                boost = 0.25
            elif intent_id == "PROMO_DISCOUNT_PRICING" and any(k in text_lower for k in ["promo", "coupon", "discount", "gift card", "deal", "pricing"]):
                boost = 0.25
            elif intent_id == "TECHNICAL_APP_WEBSITE_BUG" and any(k in text_lower for k in ["crash", "error code", "bug", "prime video", "app", "website"]):
                boost = 0.25
            elif intent_id == "FEEDBACK_AGENT_COMPLAINT" and any(k in text_lower for k in ["rude", "manager", "complaint", "worst", "unacceptable", "supervisor", "misinformation"]):
                boost = 0.25

            aggregated_scores[intent_id] = base_score + boost

        best_intent = max(aggregated_scores, key=aggregated_scores.get)
        best_score = aggregated_scores[best_intent]

        # Calibrate confidence using temperature softmax
        all_scores = np.array([aggregated_scores[i] for i in self.intent_ids])
        exp_scores = np.exp((all_scores - np.max(all_scores)) / 0.15)
        probs = exp_scores / np.sum(exp_scores)
        best_prob = float(np.max(probs))

        # Blend similarity strength with probability margin
        confidence = min(0.98, max(0.40, 0.5 * min(1.0, best_score) + 0.5 * best_prob))

        # If best raw score is very low, predict OTHER
        if best_score < 0.12:
            return "OTHER_GENERAL_INQUIRY", 0.45

        return best_intent, round(confidence, 3)

    def predict_with_details(self, text: str) -> Dict[str, Any]:
        """Returns predicted intent, confidence, and intent metadata."""
        intent, conf = self.predict(text)
        return {
            "intent": intent,
            "confidence": conf,
            "name": next((item["name"] for item in self.intents_data if item["id"] == intent), intent),
            "description": self.intent_descriptions.get(intent, ""),
            "classifier": "SemanticIntentClassifier"
        }
