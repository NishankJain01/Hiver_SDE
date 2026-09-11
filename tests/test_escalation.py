"""
Unit Tests for Escalation Engine.
"""

import pytest
from src.escalation.engine import EscalationEngine


def test_escalation_triggers_on_legal_keywords():
    engine = EscalationEngine()
    decision, reason, signals = engine.evaluate(
        customer_message="My lawyer will file a lawsuit against Amazon for fraud",
        predicted_intent="OTHER_GENERAL_INQUIRY",
        intent_confidence=0.90,
        retrieval_similarity=0.80
    )
    assert decision == "ESCALATE"
    assert signals["critical_keyword_trigger"] is True
    assert "lawyer" in signals["matched_triggers"] or "fraud" in signals["matched_triggers"]


def test_escalation_triggers_on_low_confidence():
    engine = EscalationEngine(confidence_threshold=0.65)
    decision, reason, signals = engine.evaluate(
        customer_message="I bought a blue shirt last year maybe",
        predicted_intent="RETURN_EXCHANGE_REQUEST",
        intent_confidence=0.45,  # Low confidence
        retrieval_similarity=0.75
    )
    assert decision == "ESCALATE"
    assert signals["low_intent_confidence"] is True


def test_escalation_triggers_on_high_risk_intent():
    engine = EscalationEngine()
    decision, reason, signals = engine.evaluate(
        customer_message="Help me reset my compromised account password",
        predicted_intent="ACCOUNT_ACCESS_SECURITY",
        intent_confidence=0.95,
        retrieval_similarity=0.85
    )
    assert decision == "ESCALATE"
    assert signals["high_risk_intent"] is True


def test_safe_auto_handle_decision():
    engine = EscalationEngine()
    decision, reason, signals = engine.evaluate(
        customer_message="How do I track my package under Your Orders?",
        predicted_intent="DELIVERY_SHIPPING_DELAY",
        intent_confidence=0.95,
        retrieval_similarity=0.80
    )
    assert decision == "AUTO_HANDLE"
    assert "Safe to auto-handle" in reason
