"""
Data Preparation and Cleaning Pipeline.

Processes raw or paired conversations:
- Filters for English language
- Removes handles, URLs, and agent sign-off codes
- Drops duplicates and corrupt entries
- Generates data/processed/amazon_clean_pairs.csv
- Prepares sample datasets for instant zero-download verification
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessor import clean_tweet_text, is_english_text
from src.data.reconstruct import reconstruct_brand_conversations, load_or_process_pairs
from src.data.leakage_check import detect_data_leakage, purge_evaluation_leakage


def prepare_datasets(
    output_processed_path: str = "data/processed/amazon_clean_pairs.csv",
    sample_output_path: str = "data/sample/sample_pairs.csv",
    golden_path: str = "data/golden/golden_eval.csv",
    random_seed: int = 42
):
    """Executes full data cleaning and preparation workflow."""
    print("==================================================")
    print("PHASE 1: Loading & Cleaning Brand Data (AmazonHelp)")
    print("==================================================")

    df_clean = load_or_process_pairs(
        processed_path=output_processed_path,
        ref_processed_path="_ref_repo/data/processed/amazon_pairs.csv",
        random_seed=random_seed
    )

    print(f"[INFO] Total Cleaned English Amazon Pairs: {len(df_clean):,}")

    # Create small permitted sample (500 rows) for rapid evaluation/testing
    os.makedirs(os.path.dirname(sample_output_path), exist_ok=True)
    df_sample = df_clean.sample(min(500, len(df_clean)), random_state=random_seed).copy()
    df_sample.to_csv(sample_output_path, index=False, encoding="utf-8")
    print(f"[INFO] Saved quick-test sample ({len(df_sample)} rows) to {sample_output_path}")

    # Check for leakage if golden set exists
    if os.path.exists(golden_path):
        print("\n==================================================")
        print("PHASE 2: Data Leakage Verification")
        print("==================================================")
        eval_df = pd.read_csv(golden_path, encoding="utf-8")
        leak_report = detect_data_leakage(df_clean, eval_df)
        print(f"[LEAKAGE REPORT] Clean: {leak_report['is_leakage_free']}")
        print(f"[LEAKAGE REPORT] Exact Matches: {leak_report['exact_matches_count']}")
        print(f"[LEAKAGE REPORT] Near Duplicates: {leak_report['near_matches_count']}")

        if not leak_report["is_leakage_free"]:
            print("[INFO] Purging overlapping examples from Knowledge Base to ensure 100% isolation...")
            df_isolated = purge_evaluation_leakage(df_clean, eval_df)
            df_isolated.to_csv(output_processed_path, index=False, encoding="utf-8")
            print(f"[SUCCESS] Isolated KB Saved: {len(df_isolated):,} rows.")

    print("\n[COMPLETE] Dataset preparation finished successfully.")


if __name__ == "__main__":
    prepare_datasets()
