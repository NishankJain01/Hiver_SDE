"""
Unit Tests for Agent Standardized Schema Output.
"""

import pytest
import pandas as pd
from src.agent.support_agent import SupportAgent


@pytest.fixture
def agent():
    mock_kb = pd.DataFrame({
        "tweet_id": [1, 2],
        "customer_message": [
            "Where is my package tracking info?",
            "How do I request a refund?"
        ],
        "response_text": [
            "You can track your order status in Your Orders.",
            "Refunds are processed automatically to your bank within 3-5 days."
        ]
    })
    return SupportAgent(kb_df=mock_kb)


def test_agent_output_schema_keys(agent):
    res = agent.process_message("Where is my package?")

    # Verify exact required schema keys from assignment specification
    required_keys = [
        "message",
        "intent",
        "intent_confidence",
        "retrieved_examples",
        "draft_reply",
        "decision",
        "escalation_reason"
    ]

    for key in required_keys:
        assert key in res, f"Missing required key: {key}"

    assert isinstance(res["intent"], str)
    assert isinstance(res["intent_confidence"], float)
    assert isinstance(res["retrieved_examples"], list)
    assert isinstance(res["draft_reply"], str)
    assert res["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert isinstance(res["escalation_reason"], str)
