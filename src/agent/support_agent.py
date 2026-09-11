"""
Unified Customer Support Agent Orchestrator.

End-to-end pipeline combining:
1. Input Preprocessing & Normalization
2. Intent Classification with Confidence Scoring
3. Historical Evidence Retrieval
4. Multi-Signal Escalation Engine
5. Evidence-Grounded Reply Generation
6. Standardized Output Schema Formatting
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional

from src.data.preprocessor import clean_tweet_text
from src.intents.semantic_classifier import SemanticIntentClassifier
from src.retrieval.vector_retriever import VectorRetriever
from src.escalation.engine import EscalationEngine
from src.generation.llm_client import LLMClient
from src.generation.reply_generator import ReplyGenerator


class SupportAgent:
    """
    Production-grade AI Customer Support Agent for AmazonHelp.
    """

    def __init__(
        self,
        kb_df: Optional[pd.DataFrame] = None,
        intents_config: str = "config/intents.yaml",
        settings_config: str = "config/settings.yaml",
        prompts_config: str = "config/prompts.yaml",
        llm_client: Optional[LLMClient] = None
    ):
        self.llm_client = llm_client or LLMClient()
        self.intent_classifier = SemanticIntentClassifier(
            config_path=intents_config,
            llm_client=self.llm_client
        )
        self.retriever = VectorRetriever()
        self.escalation_engine = EscalationEngine(config_path=settings_config)
        self.reply_generator = ReplyGenerator(
            llm_client=self.llm_client,
            prompts_path=prompts_config
        )

        if kb_df is not None:
            self.load_knowledge_base(kb_df)

    def load_knowledge_base(self, kb_df: pd.DataFrame):
        """Indexes the historical support knowledge base."""
        self.retriever.build_index(kb_df)

    def process_message(
        self,
        message: str,
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes an incoming customer message through the complete agent pipeline.

        Args:
            message: Raw customer message text.
            conversation_context: Optional context from prior dialogue turns.

        Returns:
            Standardized response dictionary satisfying assignment requirements:
            {
                "message": "...",
                "intent": "...",
                "intent_confidence": 0.00,
                "retrieved_examples": [...],
                "draft_reply": "...",
                "decision": "AUTO_HANDLE or ESCALATE",
                "escalation_reason": "..."
            }
        """
        # 1. Preprocess & Clean
        cleaned_msg = clean_tweet_text(message, mask_urls=True)

        # 2. Intent Classification
        intent, confidence = self.intent_classifier.predict(cleaned_msg)

        # 3. Knowledge Base Retrieval
        if self.retriever.is_indexed:
            retrieved_examples = self.retriever.retrieve(cleaned_msg)
            max_sim = self.retriever.get_max_similarity(retrieved_examples)
        else:
            retrieved_examples = []
            max_sim = 0.0

        # 4. Multi-Signal Escalation Evaluation
        decision, escalation_reason, signals = self.escalation_engine.evaluate(
            customer_message=cleaned_msg,
            predicted_intent=intent,
            intent_confidence=confidence,
            retrieval_similarity=max_sim,
            conversation_context=conversation_context
        )

        # 5. Grounded Reply Generation
        draft_reply = self.reply_generator.generate_reply(
            customer_message=cleaned_msg,
            predicted_intent=intent,
            decision=decision,
            escalation_reason=escalation_reason,
            retrieved_evidence=retrieved_examples,
            conversation_context=conversation_context
        )

        # 6. Construct Standardized Schema
        return {
            "message": message,
            "cleaned_message": cleaned_msg,
            "intent": intent,
            "intent_confidence": round(float(confidence), 3),
            "retrieved_examples": retrieved_examples,
            "draft_reply": draft_reply,
            "decision": decision,
            "escalation_reason": escalation_reason,
            "risk_signals": signals
        }
