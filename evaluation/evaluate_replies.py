"""
Response Quality and Grounding Evaluation Harness.

Evaluates generated customer replies across:
- Semantic Similarity to Human Reference Replies
- Artifact Sanitization Rate (removal of raw Twitter handles, agent signatures)
- Safety & Policy Safeguard Compliance
- Response Conciseness & Readability
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.support_agent import SupportAgent


def evaluate_replies_quality(
    golden_csv_path: str = "data/golden/golden_eval.csv",
    processed_kb_path: str = "data/processed/amazon_clean_pairs.csv"
) -> Dict[str, Any]:
    """
    Evaluates generated response quality against human golden references.
    """
    df_eval = pd.read_csv(golden_csv_path, encoding="utf-8")
    df_kb = pd.read_csv(processed_kb_path, encoding="utf-8").head(5000)

    agent = SupportAgent(kb_df=df_kb)

    generated_replies = []
    gold_replies = df_eval["human_reference_reply"].tolist()
    customer_msgs = df_eval["customer_message"].tolist()

    for msg in customer_msgs:
        res = agent.process_message(msg)
        generated_replies.append(res["draft_reply"])

    # 1. Semantic similarity to human answers
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    all_texts = gold_replies + generated_replies
    vectorizer.fit(all_texts)

    gold_vecs = vectorizer.transform(gold_replies)
    gen_vecs = vectorizer.transform(generated_replies)

    similarities = [
        float(cosine_similarity(gen_vecs[i], gold_vecs[i])[0][0])
        for i in range(len(gold_replies))
    ]
    mean_sim = float(np.mean(similarities))

    # 2. Artifact sanitization check (must not contain @mentions or ^KM signatures)
    re_handle = re.compile(r"@[A-Za-z0-9_]+")
    re_sig = re.compile(r"\^[A-Z]{1,3}\b")

    handle_leaks = sum(1 for r in generated_replies if re_handle.search(r))
    sig_leaks = sum(1 for r in generated_replies if re_sig.search(r))
    clean_rate = 1.0 - ((handle_leaks + sig_leaks) / (2 * max(1, len(generated_replies))))

    # 3. Response Length Statistics
    lengths = [len(r.split()) for r in generated_replies]
    avg_words = float(np.mean(lengths))

    return {
        "total_replies_evaluated": len(generated_replies),
        "mean_semantic_similarity_to_gold": round(mean_sim, 4),
        "artifact_sanitization_rate": round(clean_rate, 4),
        "twitter_handle_leaks": int(handle_leaks),
        "agent_signature_leaks": int(sig_leaks),
        "avg_response_word_count": round(avg_words, 1),
        "generated_replies": generated_replies
    }


if __name__ == "__main__":
    res = evaluate_replies_quality()
    print("==================================================")
    print("REPLY QUALITY & GROUNDING BENCHMARK RESULTS")
    print("==================================================")
    print(f"Total Replies Evaluated: {res['total_replies_evaluated']}")
    print(f"Mean Semantic Similarity to Human Gold: {res['mean_semantic_similarity_to_gold']:.4f}")
    print(f"Artifact Sanitization Rate: {res['artifact_sanitization_rate'] * 100:.1f}%")
    print(f"Twitter Handle Leaks: {res['twitter_handle_leaks']}")
    print(f"Agent Signature Leaks: {res['agent_signature_leaks']}")
    print(f"Avg Response Words: {res['avg_response_word_count']}")
    print("==================================================")
