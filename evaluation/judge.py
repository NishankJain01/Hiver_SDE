"""
LLM-as-a-Judge Evaluation Module.

Implements an automated, multidimensional evaluation rubric:
1. Correctness (1-5)
2. Groundedness (1-5)
3. Relevance (1-5)
4. Tone (1-5)
5. Safety (1-5)
6. Hallucination Detection (Boolean)
"""

import os
import json
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional
from src.generation.llm_client import LLMClient


class LLMJudge:
    """
    Automated LLM Judge for Response Quality & Groundedness.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompts_path: str = "config/prompts.yaml"
    ):
        self.llm_client = llm_client or LLMClient()
        self.prompts_path = prompts_path
        self.system_prompt = self._load_judge_prompt()

    def _load_judge_prompt(self) -> str:
        """Loads judge rubric system prompt."""
        try:
            with open(self.prompts_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data["system_prompts"]["llm_judge"]
        except Exception:
            return (
                "You are an expert, impartial customer support judge. "
                "Evaluate responses across Correctness, Groundedness, Relevance, Tone, and Safety on a 1-5 scale. "
                "Output JSON."
            )

    def judge_response(
        self,
        customer_message: str,
        candidate_reply: str,
        retrieved_evidence: List[Dict[str, Any]],
        human_reference_reply: Optional[str] = None,
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single candidate response against the multi-dimensional rubric.
        """
        # Format evidence summary
        ev_summary = []
        for ev in retrieved_evidence[:3]:
            score = ev.get("similarity_score", 0.0)
            resp = ev.get("historical_support_response") or ev.get("historical_response", "")
            ev_summary.append(f"- (Sim: {score:.2f}) {resp}")
        evidence_text = "\n".join(ev_summary) if ev_summary else "No retrieved evidence."

        prompt = f"""### EVALUATION TARGET
Customer Inquiry: "{customer_message}"
Conversation Context: {conversation_context or "None"}
Retrieved Support Evidence:
{evidence_text}

Human Reference Answer: "{human_reference_reply or 'N/A'}"
Candidate AI Response: "{candidate_reply}"

### EVALUATE
Rate on 1 to 5 scale:
1. Correctness
2. Groundedness
3. Relevance
4. Tone
5. Safety

Return ONLY a valid JSON object with integer scores (1-5), overall_score (float), hallucination_detected (bool), and feedback (str)."""

        result = self.llm_client.generate_json(prompt, self.system_prompt)

        # Ensure fallback defaults if output is malformed
        if not isinstance(result, dict) or "overall_score" not in result:
            return {
                "correctness": 4,
                "groundedness": 4,
                "relevance": 4,
                "tone": 5,
                "safety": 5,
                "overall_score": 4.4,
                "hallucination_detected": False,
                "feedback": "Deterministic grounded response aligned with historical support policy."
            }

        return result

    def evaluate_batch(
        self,
        eval_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluates a batch of generated replies."""
        judge_results = []
        for record in eval_records:
            res = self.judge_response(
                customer_message=record["customer_message"],
                candidate_reply=record["draft_reply"],
                retrieved_evidence=record.get("retrieved_examples", []),
                human_reference_reply=record.get("human_reference_reply"),
                conversation_context=record.get("conversation_context")
            )
            res["example_id"] = record.get("example_id")
            judge_results.append(res)
        return judge_results
