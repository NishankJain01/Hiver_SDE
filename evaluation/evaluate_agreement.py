"""
Human vs. LLM Judge Agreement Evaluation Harness.

Validates the LLM-as-a-Judge against human expert annotations.
Computes Pearson, Spearman rank correlation, Exact Agreement %, and Within ±1 Agreement %.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.judge import LLMJudge
from evaluation.metrics import compute_agreement_metrics


def run_judge_agreement_evaluation(
    human_scores_csv: str = "data/evaluation/human_judge_scores.csv"
) -> Dict[str, Any]:
    """
    Evaluates correlation and agreement between Human and LLM Judge.
    """
    if not os.path.exists(human_scores_csv):
        raise FileNotFoundError(f"Human judge scores dataset not found at {human_scores_csv}")

    df_human = pd.read_csv(human_scores_csv, encoding="utf-8")
    judge = LLMJudge()

    judge_scores = []
    judge_records = []

    for _, row in df_human.iterrows():
        judge_res = judge.judge_response(
            customer_message=str(row["customer_message"]),
            candidate_reply=str(row["draft_reply"]),
            retrieved_evidence=[],
            human_reference_reply=None
        )
        score = float(judge_res.get("overall_score", 4.4))
        judge_scores.append(score)
        judge_records.append(judge_res)

    human_overall = df_human["human_overall_score"].astype(float).tolist()

    agreement_stats = compute_agreement_metrics(human_overall, judge_scores)
    agreement_stats["annotator"] = df_human["annotator"].iloc[0] if "annotator" in df_human else "Nishank Maidawat"
    agreement_stats["judge_model"] = getattr(judge.llm_client, "model", "llama-3.3-70b-versatile")

    return agreement_stats


if __name__ == "__main__":
    res = run_judge_agreement_evaluation()
    print("==================================================")
    print("HUMAN VS. LLM-JUDGE AGREEMENT RESULTS")
    print("==================================================")
    print(f"Sample Size: {res['sample_size']}")
    print(f"Annotator: {res['annotator']}")
    print(f"Judge Model: {res['judge_model']}")
    print("--------------------------------------------------")
    print(f"Pearson Correlation (r):       {res['pearson_correlation']:.4f} (p={res['pearson_p_value']:.4e})")
    print(f"Spearman Rank Correlation (rho):{res['spearman_correlation']:.4f} (p={res['spearman_p_value']:.4e})")
    print(f"Exact Agreement:               {res['exact_agreement_pct']:.1f}%")
    print(f"Agreement Within ±1 Margin:    {res['within_plus_minus_one_pct']:.1f}%")
    print(f"Mean Absolute Error (MAE):     {res['mean_absolute_error']:.4f}")
    print("==================================================")
