# Dataset Directory Structure

This directory contains datasets and evaluation benchmarks for the Hiver AI Customer Support Agent:

- `golden/golden_eval.csv`: Hand-curated 200-sample Stratified Golden Evaluation Set with gold intent, action, and human reference replies.
- `evaluation/human_judge_scores.csv`: 40 human-annotated interactions used to validate the LLM-as-a-Judge correlation.
- `evaluation/evaluation_results.json`: Serialized output from the master evaluation harness (`python -m evaluation.run_all`).
- `sample/sample_pairs.csv`: Permitted lightweight sample (500 rows) allowing instant zero-download verification.
- `processed/amazon_clean_pairs.csv`: 58,168 cleaned English AmazonHelp support dialogues used as the historical retrieval knowledge base.
