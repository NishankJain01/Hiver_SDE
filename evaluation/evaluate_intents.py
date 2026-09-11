"""
Intent Classification Evaluation Harness.

Evaluates Baseline 0, Baseline 1, and Main System on the exact same Golden Evaluation Set.
Emphasizes Macro F1 due to class imbalance and nuanced boundaries.
"""

import os
import sys
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intents.tfidf_classifier import TfidfIntentClassifier
from src.intents.semantic_classifier import SemanticIntentClassifier
from evaluation.baselines import TrivialBaselineAgent, SimpleBaselineAgent
from evaluation.metrics import compute_classification_metrics


def run_intent_evaluation(
    golden_csv_path: str = "data/golden/golden_eval.csv",
    intents_config: str = "config/intents.yaml"
) -> Dict[str, Any]:
    """
    Runs intent classification benchmarking across all systems on golden eval set.
    """
    if not os.path.exists(golden_csv_path):
        raise FileNotFoundError(f"Golden evaluation set not found at {golden_csv_path}")

    df_eval = pd.read_csv(golden_csv_path, encoding="utf-8")
    y_true = df_eval["gold_intent"].tolist()
    texts = df_eval["customer_message"].tolist()

    labels = sorted(list(set(y_true)))

    # 1. Baseline 0: Trivial Majority Class
    b0_agent = TrivialBaselineAgent()
    b0_preds = [b0_agent.process_message(t)["intent"] for t in texts]
    b0_metrics = compute_classification_metrics(y_true, b0_preds, labels=labels)

    # 2. Baseline 1: TF-IDF + Logistic Regression (fit on exemplar/golden representation)
    semantic_helper = SemanticIntentClassifier(config_path=intents_config)
    # Fit TF-IDF model on corpus exemplars
    b1_clf = TfidfIntentClassifier()
    b1_clf.fit(semantic_helper.corpus_docs, semantic_helper.corpus_labels)
    b1_preds = [b1_clf.predict(t)[0] for t in texts]
    b1_metrics = compute_classification_metrics(y_true, b1_preds, labels=labels)

    # 3. Main System: Semantic Hybrid Classifier
    main_clf = SemanticIntentClassifier(config_path=intents_config)
    main_preds = [main_clf.predict(t)[0] for t in texts]
    main_metrics = compute_classification_metrics(y_true, main_preds, labels=labels)

    return {
        "baseline_0_trivial": b0_metrics,
        "baseline_1_simple_tfidf": b1_metrics,
        "main_system_semantic": main_metrics,
        "labels": labels,
        "total_eval_samples": len(df_eval)
    }


if __name__ == "__main__":
    results = run_intent_evaluation()
    print("==================================================")
    print("INTENT CLASSIFICATION BENCHMARK RESULTS")
    print("==================================================")
    print(f"Total Golden Evaluation Examples: {results['total_eval_samples']}")
    print("--------------------------------------------------")
    print(f"Baseline 0 (Trivial Majority)  - Accuracy: {results['baseline_0_trivial']['accuracy']:.4f} | Macro F1: {results['baseline_0_trivial']['macro_f1']:.4f}")
    print(f"Baseline 1 (TF-IDF + LogReg)   - Accuracy: {results['baseline_1_simple_tfidf']['accuracy']:.4f} | Macro F1: {results['baseline_1_simple_tfidf']['macro_f1']:.4f}")
    print(f"Main System (Semantic Hybrid)  - Accuracy: {results['main_system_semantic']['accuracy']:.4f} | Macro F1: {results['main_system_semantic']['macro_f1']:.4f}")
    print("==================================================")
