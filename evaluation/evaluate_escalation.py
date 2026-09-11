"""
Escalation Policy Evaluation Harness.

Evaluates Baseline 0, Baseline 1, and Main System on decision accuracy,
precision, recall, and safety-critical False Auto-Handle Rate.
"""

import os
import sys
import pandas as pd
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.support_agent import SupportAgent
from src.intents.tfidf_classifier import TfidfIntentClassifier
from src.intents.semantic_classifier import SemanticIntentClassifier
from src.retrieval.tfidf_retriever import TfidfRetriever
from evaluation.baselines import TrivialBaselineAgent, SimpleBaselineAgent
from evaluation.metrics import compute_escalation_metrics


def run_escalation_evaluation(
    golden_csv_path: str = "data/golden/golden_eval.csv",
    processed_kb_path: str = "data/processed/amazon_clean_pairs.csv"
) -> Dict[str, Any]:
    """
    Evaluates escalation policy across all three systems.
    """
    if not os.path.exists(golden_csv_path):
        raise FileNotFoundError(f"Golden evaluation set not found at {golden_csv_path}")

    df_eval = pd.read_csv(golden_csv_path, encoding="utf-8")
    y_true_actions = df_eval["gold_action"].tolist()
    texts = df_eval["customer_message"].tolist()

    # Load KB for retrieval evidence
    if os.path.exists(processed_kb_path):
        df_kb = pd.read_csv(processed_kb_path, encoding="utf-8").head(5000)
    else:
        df_kb = pd.DataFrame({"tweet_id": [1], "customer_message": ["where is order"], "response_text": ["check orders"]})

    # 1. Baseline 0: Trivial
    b0_agent = TrivialBaselineAgent()
    b0_preds = [b0_agent.process_message(t)["decision"] for t in texts]
    b0_metrics = compute_escalation_metrics(y_true_actions, b0_preds)

    # 2. Baseline 1: Simple Single Threshold
    semantic_helper = SemanticIntentClassifier()
    b1_agent = SimpleBaselineAgent()
    b1_agent.fit(semantic_helper.corpus_docs, semantic_helper.corpus_labels, df_kb)
    b1_preds = [b1_agent.process_message(t)["decision"] for t in texts]
    b1_metrics = compute_escalation_metrics(y_true_actions, b1_preds)

    # 3. Main System: Multi-Signal Engine
    main_agent = SupportAgent(kb_df=df_kb)
    main_preds = [main_agent.process_message(t)["decision"] for t in texts]
    main_metrics = compute_escalation_metrics(y_true_actions, main_preds)

    return {
        "baseline_0_trivial": b0_metrics,
        "baseline_1_simple_threshold": b1_metrics,
        "main_system_multi_signal": main_metrics,
        "total_eval_samples": len(df_eval)
    }


if __name__ == "__main__":
    results = run_escalation_evaluation()
    print("==================================================")
    print("ESCALATION POLICY BENCHMARK RESULTS")
    print("==================================================")
    print(f"Total Golden Evaluation Examples: {results['total_eval_samples']}")
    print("--------------------------------------------------")
    print(f"Baseline 0 (Trivial Auto-Handle) - Precision: {results['baseline_0_trivial']['precision']:.4f} | Recall: {results['baseline_0_trivial']['recall']:.4f} | False Auto-Handle Rate: {results['baseline_0_trivial']['false_auto_handle_rate']:.4f}")
    print(f"Baseline 1 (Simple Threshold)    - Precision: {results['baseline_1_simple_threshold']['precision']:.4f} | Recall: {results['baseline_1_simple_threshold']['recall']:.4f} | False Auto-Handle Rate: {results['baseline_1_simple_threshold']['false_auto_handle_rate']:.4f}")
    print(f"Main System (Multi-Signal Policy)- Precision: {results['main_system_multi_signal']['precision']:.4f} | Recall: {results['main_system_multi_signal']['recall']:.4f} | False Auto-Handle Rate: {results['main_system_multi_signal']['false_auto_handle_rate']:.4f}")
    print("==================================================")
