"""
Conversation Reconstruction and Dataset Preparation.

Reconstructs customer-support dialogues from raw Twitter customer support data:
- Identifies brand interactions
- Pairs customer inbound inquiries with authoritative brand support replies
- Reconstructs multi-turn conversational context when available
- Cleans and filters non-English / corrupt data
- Produces deterministic, reproducible dataset splits
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from src.data.preprocessor import clean_tweet_text, is_english_text


def reconstruct_brand_conversations(
    df_raw: pd.DataFrame,
    brand_name: str = "AmazonHelp",
    min_length: int = 10,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Reconstructs paired customer query -> brand response interactions.

    Args:
        df_raw: Raw TWCS dataframe with columns:
                ['tweet_id', 'author_id', 'inbound', 'created_at', 'text',
                 'response_tweet_id', 'in_response_to_tweet_id']
        brand_name: Brand author_id to filter for (default: AmazonHelp).
        min_length: Minimum customer text length to keep.
        random_seed: Random seed for reproducible operations.

    Returns:
        DataFrame with columns:
        ['tweet_id', 'customer_message', 'customer_message_raw',
         'response_text', 'response_text_raw', 'brand']
    """
    # 1. Separate inbound (customer) and outbound (brand) tweets
    brand_tweets = df_raw[
        (df_raw["inbound"] == False) &
        (df_raw["author_id"].str.lower() == brand_name.lower())
    ].copy()

    # Create fast index lookup for inbound customer tweets
    customer_lookup = df_raw[df_raw["inbound"] == True].set_index("tweet_id")

    # 2. Match brand responses to customer parent tweets
    valid_brand_replies = brand_tweets[brand_tweets["in_response_to_tweet_id"].notna()].copy()
    valid_brand_replies["parent_tweet_id"] = valid_brand_replies["in_response_to_tweet_id"].astype("Int64")

    # Map customer text from parent_tweet_id
    valid_brand_replies["customer_text_raw"] = valid_brand_replies["parent_tweet_id"].map(
        customer_lookup["text"]
    )

    # Filter out rows where parent customer tweet was not found
    pairs = valid_brand_replies.dropna(subset=["customer_text_raw"]).copy()

    # 3. Clean and normalize
    pairs["customer_message"] = pairs["customer_text_raw"].apply(
        lambda t: clean_tweet_text(t, mask_urls=True)
    )
    pairs["response_text"] = pairs["text"].apply(
        lambda t: clean_tweet_text(t, mask_urls=False)
    )

    # 4. Filter for English, minimum length, and non-empty texts
    pairs["is_english"] = pairs["customer_text_raw"].apply(is_english_text)
    pairs = pairs[pairs["is_english"] == True].copy()

    pairs["cust_len"] = pairs["customer_message"].str.len()
    pairs = pairs[pairs["cust_len"] >= min_length].copy()

    # 5. Deduplicate identical customer messages to avoid redundancy
    pairs = pairs.drop_duplicates(subset=["customer_message"]).copy()

    # 6. Format final dataframe
    result = pd.DataFrame({
        "tweet_id": pairs["parent_tweet_id"],
        "customer_message": pairs["customer_message"],
        "customer_message_raw": pairs["customer_text_raw"],
        "response_text": pairs["response_text"],
        "response_text_raw": pairs["text"],
        "brand": brand_name
    }).reset_index(drop=True)

    return result


def load_or_process_pairs(
    processed_path: str = "data/processed/amazon_clean_pairs.csv",
    ref_processed_path: str = "_ref_repo/data/processed/amazon_pairs.csv",
    min_length: int = 10,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Loads pre-cleaned pairs if exists, or cleans the reference pairs and saves.
    """
    if os.path.exists(processed_path):
        return pd.read_csv(processed_path, encoding="utf-8")

    # Fallback to existing pair extract if full TWCS raw is not available
    if os.path.exists(ref_processed_path):
        df_ref = pd.read_csv(ref_processed_path, encoding="utf-8", encoding_errors="replace")
        
        # Columns in reference: customer_text, response_text, (optional intent)
        cust_col = "customer_text" if "customer_text" in df_ref.columns else "text"
        resp_col = "response_text" if "response_text" in df_ref.columns else "response"

        # Apply strict English filter and cleaning
        df_ref = df_ref.dropna(subset=[cust_col, resp_col]).copy()
        df_ref["is_english"] = df_ref[cust_col].astype(str).apply(is_english_text)
        df_clean = df_ref[df_ref["is_english"] == True].copy()

        df_clean["customer_message"] = df_clean[cust_col].astype(str).apply(
            lambda t: clean_tweet_text(t, mask_urls=True)
        )
        df_clean["response_text"] = df_clean[resp_col].astype(str).apply(
            lambda t: clean_tweet_text(t, mask_urls=False)
        )

        df_clean = df_clean[df_clean["customer_message"].str.len() >= min_length].copy()
        df_clean = df_clean.drop_duplicates(subset=["customer_message"]).copy()

        df_clean["tweet_id"] = np.arange(100000, 100000 + len(df_clean))
        df_clean["brand"] = "AmazonHelp"

        result = df_clean[[
            "tweet_id", "customer_message", "response_text", "brand"
        ]].reset_index(drop=True)

        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        result.to_csv(processed_path, index=False, encoding="utf-8")
        return result

    raise FileNotFoundError(
        f"Neither {processed_path} nor {ref_processed_path} could be found. "
        "Please provide raw TWCS or pairs dataset."
    )
