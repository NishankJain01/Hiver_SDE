"""
Evaluation Framework Package.
Provides Baselines, Intent Metrics, Escalation Metrics, Reply Evaluation,
LLM-as-Judge, and Human-Judge Agreement Harness.
"""

from evaluation.metrics import compute_classification_metrics, compute_escalation_metrics, compute_agreement_metrics
from evaluation.baselines import TrivialBaselineAgent, SimpleBaselineAgent

__all__ = [
    "compute_classification_metrics",
    "compute_escalation_metrics",
    "compute_agreement_metrics",
    "TrivialBaselineAgent",
    "SimpleBaselineAgent",
]
