"""
Rule-Based Keyword Intent Classifier.

Implements heuristic rule matching as a deterministic baseline.
"""

import re
from typing import Dict, Any, Tuple


class RuleIntentClassifier:
    """
    Keyword and regex-based intent classifier.
    """

    INTENT_KEYWORDS = {
        "REFUND_BILLING_INQUIRY": [
            "refund", "refunded", "refunding", "money back", "double charged",
            "overcharged", "unauthorized charge", "bank debit", "credit card charge",
            "billing statement", "invoice charge", "deducted twice", "payment dispute"
        ],
        "RETURN_EXCHANGE_REQUEST": [
            "return", "returning", "returns", "exchange", "replacement",
            "return label", "pickup", "drop off", "return window", "send back"
        ],
        "DELIVERY_SHIPPING_DELAY": [
            "delivery", "delivered", "shipping", "package", "parcel", "courier",
            "tracking", "shipped", "carrier", "delayed", "where is my order",
            "late arrival", "tracking number", "attempted delivery", "ontrac", "usps", "ups"
        ],
        "DAMAGED_DEFECTIVE_PRODUCT": [
            "damaged", "broken", "defective", "faulty", "shattered", "cracked",
            "missing parts", "wrong item", "different size sent", "dented", "scratched",
            "does not work", "malfunctioning"
        ],
        "ACCOUNT_ACCESS_SECURITY": [
            "login", "log in", "sign in", "account locked", "password reset",
            "otp", "two factor", "2fa", "verification code", "hacked",
            "compromised", "unauthorized login", "membership access"
        ],
        "ORDER_CANCELLATION_MODIFICATION": [
            "cancel order", "cancel my", "cancellation", "change address",
            "modify order", "wrong address", "update delivery address", "stop shipment"
        ],
        "PROMO_DISCOUNT_PRICING": [
            "promo code", "coupon", "discount", "gift card", "claim code",
            "voucher", "price match", "lightning deal", "cashback", "promotional balance"
        ],
        "TECHNICAL_APP_WEBSITE_BUG": [
            "app crash", "website error", "checkout error", "page not loading",
            "error 500", "prime video error", "streaming bug", "kindle sync",
            "app not working", "site down"
        ],
        "FEEDBACK_AGENT_COMPLAINT": [
            "rude agent", "complaint against", "terrible service", "useless support",
            "unacceptable service", "worst customer care", "supervisor", "manager callback",
            "nobody is helping", "broken promise"
        ]
    }

    def __init__(self):
        pass

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predicts intent and confidence based on keyword matching score.

        Returns:
            Tuple of (predicted_intent_id, confidence_score)
        """
        if not text or not isinstance(text, str):
            return "OTHER_GENERAL_INQUIRY", 0.30

        text_lower = text.lower()
        best_intent = "OTHER_GENERAL_INQUIRY"
        best_score = 0.0

        for intent, keywords in self.INTENT_KEYWORDS.items():
            matches = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text_lower))
            if matches > 0:
                # Score based on matches and specificity
                score = min(0.95, 0.50 + matches * 0.20)
                if score > best_score:
                    best_score = score
                    best_intent = intent

        if best_score == 0.0:
            return "OTHER_GENERAL_INQUIRY", 0.40

        return best_intent, round(best_score, 3)

    def predict_with_details(self, text: str) -> Dict[str, Any]:
        """Predicts intent with reasoning metadata."""
        intent, conf = self.predict(text)
        return {
            "intent": intent,
            "confidence": conf,
            "classifier": "RuleIntentClassifier",
            "method": "keyword_matching"
        }
