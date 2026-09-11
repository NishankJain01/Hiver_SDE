"""
Grounded Reply Generator Module.

Constructs strict anti-hallucination prompts injecting retrieved evidence
and generates brand-grounded suggested support responses.
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from src.generation.llm_client import LLMClient


class ReplyGenerator:
    """
    Evidence-Grounded Support Reply Generator.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompts_path: str = "config/prompts.yaml"
    ):
        self.llm_client = llm_client or LLMClient()
        self.prompts_path = prompts_path
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Loads generation system prompt from configuration."""
        try:
            with open(self.prompts_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data["system_prompts"]["reply_generation"]
        except Exception:
            return (
                "You are an expert, empathetic, and grounded Customer Support AI Assistant for Amazon. "
                "Respond strictly based on the retrieved historical support evidence. Never invent policies or promises."
            )

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        decision: str,
        escalation_reason: str,
        retrieved_evidence: List[Dict[str, Any]],
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Synthesizes a grounded customer support reply.

        Args:
            customer_message: Incoming customer inquiry.
            predicted_intent: Classified intent.
            decision: "AUTO_HANDLE" or "ESCALATE".
            escalation_reason: Reason for decision.
            retrieved_evidence: Top-k historical support interactions.
            conversation_context: Prior dialogue turns if available.

        Returns:
            Draft response string.
        """
        # Format evidence block
        evidence_lines = []
        for ev in retrieved_evidence:
            rank = ev.get("rank", 1)
            score = ev.get("similarity_score", 0.0)
            hist_cust = ev.get("historical_customer_message", "")
            hist_resp = ev.get("historical_support_response") or ev.get("historical_response", "")
            evidence_lines.append(
                f"[Evidence #{rank} | Similarity: {score:.2f}]\n"
                f"Customer: {hist_cust}\n"
                f"Amazon Support: {hist_resp}"
            )
        evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "No relevant historical evidence found."

        # Construct generation prompt
        prompt = f"""### CUSTOMER INQUIRY
Customer Message: "{customer_message}"
Conversation Context: {conversation_context or "None (New conversation)"}

### CLASSIFICATION & DECISION
Predicted Intent: {predicted_intent}
Decision: {decision}
Decision Reason: {escalation_reason}

### RETRIEVED HISTORICAL EVIDENCE (GROUND TRUTH CONTEXT)
{evidence_block}

### INSTRUCTIONS
Generate a concise, empathetic, and grounded support reply (2-4 sentences).
- If Decision is AUTO_HANDLE: Use the historical support evidence to address the customer's problem directly.
- If Decision is ESCALATE: Acknowledge the issue empathetically and inform the customer that a human specialist will review their case.
- DO NOT invent refund amounts, specific delivery guarantees, or account actions not present in the evidence.
- DO NOT use @mentions or agent sign-off codes (like ^KM).

Draft Suggested Response:"""

        return self.llm_client.generate(prompt, self.system_prompt)
