"""
Evaluation Metrics Computation Module.

Computes mathematically rigorous metrics:
- Multiclass Classification: Accuracy, Macro F1, Weighted F1, Per-Class Precision/Recall/F1
- Binary Escalation: Precision, Recall, F1, False Auto-Handle Rate (Safety Critical), Confusion Matrix
- Judge Agreement: Pearson Correlation, Spearman Rank Correlation, Exact Agreement, Within ±1 Agreement
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from scipy.stats import pearsonr, spearmanr


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive intent classification metrics.
    """
    if labels is None:
        labels = sorted(list(set(y_true).union(set(y_pred))))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    # Per-class metrics
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for label in labels:
        if label in report:
            per_class[label] = {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1_score": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"])
            }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def compute_escalation_metrics(
    y_true_actions: List[str],
    y_pred_actions: List[str]
) -> Dict[str, Any]:
    """
    Evaluates escalation policy performance.
    Treats ESCALATE as the positive class (1) and AUTO_HANDLE as (0).

    Computes False Auto-Handle Rate: P(Pred=AUTO_HANDLE | True=ESCALATE),
    representing cases where human intervention was required but the system
    falsely attempted automation.
    """
    # Normalize to binary (1 = ESCALATE, 0 = AUTO_HANDLE)
    bin_true = [1 if a.upper() == "ESCALATE" else 0 for a in y_true_actions]
    bin_pred = [1 if a.upper() == "ESCALATE" else 0 for a in y_pred_actions]

    acc = accuracy_score(bin_true, bin_pred)
    prec = precision_score(bin_true, bin_pred, zero_division=0)
    rec = recall_score(bin_true, bin_pred, zero_division=0)
    f1 = f1_score(bin_true, bin_pred, zero_division=0)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    # TN: True Auto-Handle, FP: Unnecessary Escalation
    # FN: Dangerous False Auto-Handle, TP: Correct Escalation
    cm = confusion_matrix(bin_true, bin_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Safety critical metric: False Auto-Handle Rate = FN / (FN + TP)
    total_escalated_gold = fn + tp
    false_auto_handle_rate = (fn / total_escalated_gold) if total_escalated_gold > 0 else 0.0

    # Unnecessary Escalation Rate = FP / (TN + FP)
    total_autohandle_gold = tn + fp
    unnecessary_escalate_rate = (fp / total_autohandle_gold) if total_autohandle_gold > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "true_positives_correct_escalations": int(tp),
        "true_negatives_correct_autohandles": int(tn),
        "false_positives_unnecessary_escalations": int(fp),
        "false_negatives_dangerous_auto_handles": int(fn),
        "false_auto_handle_rate": round(float(false_auto_handle_rate), 4),
        "unnecessary_escalate_rate": round(float(unnecessary_escalate_rate), 4),
        "confusion_matrix": {
            "tn_auto_handle": int(tn),
            "fp_unnecessary_escalation": int(fp),
            "fn_false_auto_handle": int(fn),
            "tp_correct_escalation": int(tp)
        }
    }


def compute_agreement_metrics(
    human_scores: List[float],
    judge_scores: List[float]
) -> Dict[str, Any]:
    """
    Computes correlation and agreement metrics between Human and LLM Judge.
    """
    h = np.array(human_scores, dtype=float)
    j = np.array(judge_scores, dtype=float)

    if len(h) < 2:
        return {"error": "Insufficient sample size for correlation analysis."}

    # Pearson and Spearman
    p_corr, p_val = pearsonr(h, j)
    s_corr, s_val = spearmanr(h, j)

    # Exact agreement and within +- 1 margin
    exact_matches = np.sum(np.round(h) == np.round(j))
    exact_pct = exact_matches / len(h)

    within_one = np.sum(np.abs(h - j) <= 1.0)
    within_one_pct = within_one / len(h)

    mae = np.mean(np.abs(h - j))

    return {
        "sample_size": len(h),
        "pearson_correlation": round(float(p_corr), 4),
        "pearson_p_value": round(float(p_val), 6),
        "spearman_correlation": round(float(s_corr), 4),
        "spearman_p_value": round(float(s_val), 6),
        "exact_agreement_pct": round(float(exact_pct * 100), 2),
        "within_plus_minus_one_pct": round(float(within_one_pct * 100), 2),
        "mean_absolute_error": round(float(mae), 4)
    }
