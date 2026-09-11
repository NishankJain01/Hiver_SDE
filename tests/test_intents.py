"""
Unit Tests for Intent Classifiers.
"""

import pytest
from src.intents.rule_classifier import RuleIntentClassifier
from src.intents.semantic_classifier import SemanticIntentClassifier


def test_rule_classifier_predictions():
    clf = RuleIntentClassifier()

    intent, conf = clf.predict("I need a refund for my double charged credit card")
    assert intent == "REFUND_BILLING_INQUIRY"
    assert conf > 0.50

    intent, conf = clf.predict("Where is my package? The tracking has been delayed")
    assert intent == "DELIVERY_SHIPPING_DELAY"
    assert conf > 0.50

    intent, conf = clf.predict("The glass bowl arrived completely broken and shattered")
    assert intent == "DAMAGED_DEFECTIVE_PRODUCT"
    assert conf > 0.50


def test_semantic_classifier_predictions():
    clf = SemanticIntentClassifier()

    intent, conf = clf.predict("Someone logged into my account and changed the 2FA password")
    assert intent == "ACCOUNT_ACCESS_SECURITY"
    assert conf >= 0.50

    intent, conf = clf.predict("I want to exchange this shirt for a size Large")
    assert intent == "RETURN_EXCHANGE_REQUEST"
    assert conf >= 0.50

    intent, conf = clf.predict("Your chat agent was extremely rude and unhelpful")
    assert intent == "FEEDBACK_AGENT_COMPLAINT"
    assert conf >= 0.50
