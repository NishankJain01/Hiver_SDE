"""
Unit Tests for Data Leakage Detection.
"""

import pytest
import pandas as pd
from src.data.leakage_check import detect_data_leakage, purge_evaluation_leakage


def test_detect_data_leakage_clean():
    kb = pd.DataFrame({"customer_message": ["Where is my order", "I want to return shoes"]})
    eval_df = pd.DataFrame({"customer_message": ["How to reset password", "Double charged on Prime"]})

    report = detect_data_leakage(kb, eval_df)
    assert report["is_leakage_free"] is True
    assert report["exact_matches_count"] == 0


def test_detect_data_leakage_contaminated():
    kb = pd.DataFrame({"customer_message": ["Where is my order", "I want to return shoes"]})
    eval_df = pd.DataFrame({"customer_message": ["Where is my order", "How to cancel subscription"]})

    report = detect_data_leakage(kb, eval_df)
    assert report["is_leakage_free"] is False
    assert report["exact_matches_count"] == 1


def test_purge_evaluation_leakage():
    kb = pd.DataFrame({"customer_message": ["Where is my order", "I want to return shoes", "Unique KB item"]})
    eval_df = pd.DataFrame({"customer_message": ["Where is my order", "I want to return shoes"]})

    clean_kb = purge_evaluation_leakage(kb, eval_df)
    assert len(clean_kb) == 1
    assert clean_kb.iloc[0]["customer_message"] == "Unique KB item"
