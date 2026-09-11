"""
Data Leakage and Overlap Detection Module.

Prevents contamination between the retrieval knowledge base and the golden evaluation set.
Ensures evaluation credibility by identifying exact and near-duplicate leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from src.data.preprocessor import clean_tweet_text


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Computes word-level Jaccard similarity between two text strings."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def detect_data_leakage(
    kb_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    kb_text_col: str = "customer_message",
    eval_text_col: str = "customer_message",
    similarity_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Checks for exact and near-duplicate leakage between retrieval KB and evaluation set.

    Args:
        kb_df: Knowledge base dataframe used for retrieval.
        eval_df: Golden evaluation set dataframe.
        kb_text_col: Column name for KB text.
        eval_text_col: Column name for evaluation text.
        similarity_threshold: Threshold above which two queries are flagged as near-duplicates.

    Returns:
        Dictionary containing leakage statistics and list of contaminated examples.
    """
    kb_texts = set(kb_df[kb_text_col].dropna().astype(str).str.strip().str.lower())
    eval_texts = eval_df[eval_text_col].dropna().astype(str).str.strip().tolist()

    exact_matches = []
    near_matches = []

    for idx, eval_text in enumerate(eval_texts):
        cleaned_eval = eval_text.lower()
        # 1. Exact match check
        if cleaned_eval in kb_texts:
            exact_matches.append({
                "eval_index": idx,
                "eval_text": eval_text,
                "match_type": "EXACT"
            })
            continue

        # 2. Near-duplicate sample check
        # For performance, only check subset or candidate tokens
        eval_tokens = set(cleaned_eval.split())
        for kb_text in kb_texts:
            kb_tokens = set(kb_text.split())
            if not kb_tokens:
                continue
            # Quick filter: length ratio
            if abs(len(eval_tokens) - len(kb_tokens)) > 5:
                continue
            jaccard = len(eval_tokens.intersection(kb_tokens)) / len(eval_tokens.union(kb_tokens))
            if jaccard >= similarity_threshold:
                near_matches.append({
                    "eval_index": idx,
                    "eval_text": eval_text,
                    "kb_text": kb_text,
                    "jaccard_similarity": round(jaccard, 3),
                    "match_type": "NEAR_DUPLICATE"
                })
                break

    is_clean = (len(exact_matches) == 0 and len(near_matches) == 0)

    return {
        "is_leakage_free": is_clean,
        "total_eval_samples": len(eval_df),
        "total_kb_samples": len(kb_df),
        "exact_matches_count": len(exact_matches),
        "near_matches_count": len(near_matches),
        "exact_matches": exact_matches,
        "near_matches": near_matches,
        "leakage_rate": round((len(exact_matches) + len(near_matches)) / max(1, len(eval_df)), 4)
    }


def purge_evaluation_leakage(
    kb_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    kb_text_col: str = "customer_message",
    eval_text_col: str = "customer_message"
) -> pd.DataFrame:
    """
    Removes any examples from the Knowledge Base that match the evaluation set.
    Ensures absolute data isolation.
    """
    eval_texts_normalized = set(
        eval_df[eval_text_col].dropna().astype(str).str.strip().str.lower()
    )
    
    clean_kb = kb_df[
        ~kb_df[kb_text_col].astype(str).str.strip().str.lower().isin(eval_texts_normalized)
    ].copy().reset_index(drop=True)

    return clean_kb
