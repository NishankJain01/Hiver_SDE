"""
Master Evaluation Harness.

Runs the complete evaluation suite across all systems and benchmarks:
1. Intent Classification (Baseline 0 vs Baseline 1 vs Main System)
2. Escalation Policy (Decision Accuracy, False Auto-Handle Rate)
3. Reply Quality & Grounding
4. LLM-as-a-Judge Evaluation (5 Dimensions)
5. Human vs. LLM-Judge Correlation & Agreement Analysis

Serializes all empirical results to data/evaluation/evaluation_results.json.
Execution completes in < 5 minutes on standard hardware.
"""

import os
import sys
import json
import time
from typing import Dict, Any

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.evaluate_intents import run_intent_evaluation
from evaluation.evaluate_escalation import run_escalation_evaluation
from evaluation.evaluate_replies import evaluate_replies_quality
from evaluation.evaluate_agreement import run_judge_agreement_evaluation


def run_complete_evaluation(
    golden_csv: str = "data/golden/golden_eval.csv",
    kb_csv: str = "data/processed/amazon_clean_pairs.csv",
    human_scores_csv: str = "data/evaluation/human_judge_scores.csv",
    output_json: str = "data/evaluation/evaluation_results.json"
) -> Dict[str, Any]:
    """
    Executes master evaluation pipeline.
    """
    start_time = time.time()
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    print("================================================================================")
    print("           HIVER AI CUSTOMER SUPPORT AGENT - MASTER BENCHMARK RUNNER            ")
    print("           Candidate: Nishank Maidawat | Brand: AmazonHelp (TWCS)                ")
    print("================================================================================")

    # 1. Intent Classification
    print("\n[1/4] Running Multiclass Intent Classification Benchmark...")
    intent_results = run_intent_evaluation(golden_csv_path=golden_csv)
    print(f"      -> Baseline 0 (Trivial Majority): Acc={intent_results['baseline_0_trivial']['accuracy']:.4f} | Macro F1={intent_results['baseline_0_trivial']['macro_f1']:.4f}")
    print(f"      -> Baseline 1 (TF-IDF + LogReg):  Acc={intent_results['baseline_1_simple_tfidf']['accuracy']:.4f} | Macro F1={intent_results['baseline_1_simple_tfidf']['macro_f1']:.4f}")
    print(f"      -> Main System (Semantic Hybrid): Acc={intent_results['main_system_semantic']['accuracy']:.4f} | Macro F1={intent_results['main_system_semantic']['macro_f1']:.4f}")

    # 2. Escalation Policy Evaluation
    print("\n[2/4] Running Escalation Policy Benchmark (Auto-Handle vs. Escalate)...")
    escalation_results = run_escalation_evaluation(golden_csv_path=golden_csv, processed_kb_path=kb_csv)
    print(f"      -> Baseline 0 (Trivial Default): Prec={escalation_results['baseline_0_trivial']['precision']:.4f} | False Auto-Handle Rate={escalation_results['baseline_0_trivial']['false_auto_handle_rate']:.4f}")
    print(f"      -> Baseline 1 (Simple Threshold): Prec={escalation_results['baseline_1_simple_threshold']['precision']:.4f} | False Auto-Handle Rate={escalation_results['baseline_1_simple_threshold']['false_auto_handle_rate']:.4f}")
    print(f"      -> Main System (Multi-Signal):   Prec={escalation_results['main_system_multi_signal']['precision']:.4f} | False Auto-Handle Rate={escalation_results['main_system_multi_signal']['false_auto_handle_rate']:.4f}")

    # 3. Reply Quality & Grounding
    print("\n[3/4] Running Response Generation & Evidence Grounding Benchmark...")
    reply_results = evaluate_replies_quality(golden_csv_path=golden_csv, processed_kb_path=kb_csv)
    print(f"      -> Mean Semantic Similarity to Human Gold: {reply_results['mean_semantic_similarity_to_gold']:.4f}")
    print(f"      -> Twitter Handle & Artifact Sanitization: {reply_results['artifact_sanitization_rate'] * 100:.1f}%")
    print(f"      -> Average Generated Word Count: {reply_results['avg_response_word_count']} words")

    # 4. Human vs. LLM Judge Agreement
    print("\n[4/4] Running Human vs. LLM-as-a-Judge Agreement Validation...")
    agreement_results = run_judge_agreement_evaluation(human_scores_csv=human_scores_csv)
    print(f"      -> Sample Size: {agreement_results['sample_size']} Human Evaluated Interactions")
    print(f"      -> Pearson Correlation (r):        {agreement_results['pearson_correlation']:.4f}")
    print(f"      -> Spearman Rank Correlation (rho): {agreement_results['spearman_correlation']:.4f}")
    print(f"      -> Agreement Within ±1 Margin:     {agreement_results['within_plus_minus_one_pct']:.1f}%")

    elapsed_time = round(time.time() - start_time, 2)

    master_results = {
        "metadata": {
            "candidate": "Nishank Maidawat",
            "project": "Hiver AI Support Agent",
            "brand": "AmazonHelp",
            "benchmark_dataset": "Golden Evaluation Set (200 curated examples)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "execution_time_seconds": elapsed_time
        },
        "intent_classification": intent_results,
        "escalation_policy": escalation_results,
        "reply_quality": {
            "total_evaluated": reply_results["total_replies_evaluated"],
            "mean_semantic_similarity_to_gold": reply_results["mean_semantic_similarity_to_gold"],
            "artifact_sanitization_rate": reply_results["artifact_sanitization_rate"],
            "avg_response_word_count": reply_results["avg_response_word_count"]
        },
        "human_vs_judge_agreement": agreement_results
    }

    # Save to disk
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print("\n================================================================================")
    print(f"[SUCCESS] Master Benchmark Suite Completed in {elapsed_time} seconds!")
    print(f"[SAVED] Results serialized to: {os.path.abspath(output_json)}")
    print("================================================================================")

    # Print Summary Markdown Table for Evaluator
    print("\n### SUMMARY COMPARISON TABLE FOR EVALUATORS\n")
    print("| System | Intent Accuracy | Intent Macro F1 | Escalation Prec | Escalation Rec | False Auto-Handle Rate (Safety) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(f"| **Baseline 0 (Trivial)** | {intent_results['baseline_0_trivial']['accuracy']:.3f} | {intent_results['baseline_0_trivial']['macro_f1']:.3f} | {escalation_results['baseline_0_trivial']['precision']:.3f} | {escalation_results['baseline_0_trivial']['recall']:.3f} | {escalation_results['baseline_0_trivial']['false_auto_handle_rate']:.3f} |")
    print(f"| **Baseline 1 (Simple TF-IDF)** | {intent_results['baseline_1_simple_tfidf']['accuracy']:.3f} | {intent_results['baseline_1_simple_tfidf']['macro_f1']:.3f} | {escalation_results['baseline_1_simple_threshold']['precision']:.3f} | {escalation_results['baseline_1_simple_threshold']['recall']:.3f} | {escalation_results['baseline_1_simple_threshold']['false_auto_handle_rate']:.3f} |")
    print(f"| **Main System (Semantic + Multi-Signal)** | **{intent_results['main_system_semantic']['accuracy']:.3f}** | **{intent_results['main_system_semantic']['macro_f1']:.3f}** | **{escalation_results['main_system_multi_signal']['precision']:.3f}** | **{escalation_results['main_system_multi_signal']['recall']:.3f}** | **{escalation_results['main_system_multi_signal']['false_auto_handle_rate']:.3f}** |")

    return master_results


if __name__ == "__main__":
    run_complete_evaluation()
