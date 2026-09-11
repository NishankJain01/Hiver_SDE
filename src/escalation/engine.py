"""
Multi-Signal Escalation Engine.

Evaluates multiple risk signals to decide whether an incoming customer inquiry
can be safely auto-handled or must be escalated to a human specialist.

Signals Evaluated:
1. Intent Classification Confidence (< threshold)
2. Retrieval Evidence Similarity (< threshold)
3. Sensitive & High-Risk Intent Policies (Security, Complaints)
4. Critical Security, Legal, Fraud, and Safety Triggers
5. High Customer Frustration & Repetitive Failure Indicators
"""

import re
import yaml
from typing import Dict, Any, List, Optional, Tuple


class EscalationEngine:
    """
    Explainable Multi-Signal Escalation Engine.
    """

    DEFAULT_ALWAYS_ESCALATE = [
        "lawyer", "attorney", "legal action", "sue you", "sue amazon",
        "police", "stolen card", "unauthorized access", "hacked",
        "fraud", "scam", "consumer court", "bbb complaint", "fbi",
        "threat", "manager immediately", "unacceptable behavior",
        "safety hazard", "fire", "electrical shock", "injury"
    ]

    HIGH_RISK_INTENTS = [
        "ACCOUNT_ACCESS_SECURITY",
        "FEEDBACK_AGENT_COMPLAINT"
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        retrieval_threshold: float = 0.55,
        config_path: Optional[str] = "config/settings.yaml"
    ):
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold
        self.always_escalate_keywords = self.DEFAULT_ALWAYS_ESCALATE
        self.high_risk_intents = self.HIGH_RISK_INTENTS

        if config_path:
            self._load_from_config(config_path)

    def _load_from_config(self, config_path: str):
        """Loads thresholds and risk keywords from YAML settings."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)
            esc_cfg = settings.get("escalation", {})
            self.confidence_threshold = esc_cfg.get("low_confidence_threshold", self.confidence_threshold)
            self.retrieval_threshold = esc_cfg.get("weak_retrieval_threshold", self.retrieval_threshold)
            self.high_risk_intents = esc_cfg.get("high_risk_intents", self.high_risk_intents)
            self.always_escalate_keywords = esc_cfg.get("always_escalate_keywords", self.always_escalate_keywords)
        except Exception:
            pass  # Retain defaults if config is missing

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieval_similarity: float,
        conversation_context: Optional[str] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Evaluates incoming inquiry across multiple risk dimensions.

        Args:
            customer_message: Cleaned customer message.
            predicted_intent: Intent predicted by classifier.
            intent_confidence: Calibrated intent confidence score (0.0 to 1.0).
            retrieval_similarity: Top cosine similarity score from historical KB.
            conversation_context: Optional multi-turn history.

        Returns:
            Tuple of (decision, reason, structured_signals_dict)
            decision: "AUTO_HANDLE" or "ESCALATE"
        """
        msg_lower = (customer_message or "").lower()
        reasons: List[str] = []
        signals = {
            "low_intent_confidence": False,
            "weak_retrieval_evidence": False,
            "high_risk_intent": False,
            "critical_keyword_trigger": False,
            "ambiguous_or_empty": False,
            "matched_triggers": []
        }

        # Check 0: Empty or extremely short query
        if len(msg_lower.strip()) < 5:
            signals["ambiguous_or_empty"] = True
            reasons.append("Customer inquiry is too short or ambiguous to construct a safe response.")

        # Check 1: Critical safety / legal / security keyword patterns
        matched_kws = []
        for kw in self.always_escalate_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", msg_lower):
                matched_kws.append(kw)

        if matched_kws:
            signals["critical_keyword_trigger"] = True
            signals["matched_triggers"] = matched_kws
            reasons.append(f"Contains high-risk triggers ({', '.join(matched_kws[:3])}) requiring human specialist review.")

        # Check 2: Sensitive intent policies
        if predicted_intent in self.high_risk_intents:
            signals["high_risk_intent"] = True
            if predicted_intent == "ACCOUNT_ACCESS_SECURITY":
                reasons.append("Account security and authentication issues require verified human identity workflows.")
            elif predicted_intent == "FEEDBACK_AGENT_COMPLAINT":
                reasons.append("Customer feedback or agent dissatisfaction requires personalized human resolution.")

        # Check 3: Intent Classification Confidence
        if intent_confidence < self.confidence_threshold:
            signals["low_intent_confidence"] = True
            reasons.append(
                f"Low intent confidence ({intent_confidence:.2f} < threshold {self.confidence_threshold:.2f})."
            )

        # Check 4: Retrieval Evidence Grounding Quality
        if retrieval_similarity < self.retrieval_threshold:
            signals["weak_retrieval_evidence"] = True
            reasons.append(
                f"Insufficient historical evidence grounding (similarity {retrieval_similarity:.2f} < threshold {self.retrieval_threshold:.2f})."
            )

        # Final Decision Synthesis
        if len(reasons) > 0:
            decision = "ESCALATE"
            primary_reason = " | ".join(reasons)
        else:
            decision = "AUTO_HANDLE"
            primary_reason = (
                f"High intent confidence ({intent_confidence:.2f}) and strong historical evidence grounding "
                f"(similarity {retrieval_similarity:.2f}). Safe to auto-handle."
            )

        return decision, primary_reason, signals
