# Failure Analysis: Top 5 Real Failure Modes

**Project:** Hiver AI Customer Support Agent  
**Author:** Nishank Maidawat  
**Analysis Date:** September 2026  
**Empirical Source:** Benchmark evaluation runs on `data/golden/golden_eval.csv`

---

## Overview

Rather than hypothesizing theoretical weaknesses, this document analyzes **5 real failure modes** observed during empirical benchmarking of our intent classification, vector retrieval, and escalation pipeline. For each failure mode, we provide an anonymized dataset example, root cause analysis, and actionable remediation steps.

---

## 1. Failure Mode 1: Intent Ambiguity in Multi-Barrier Compound Queries

### Dataset Example
- **Customer Message:** *"My package was delayed by 2 weeks so I want to return it and get an immediate refund on my card."*
- **Gold Label:** `REFUND_BILLING_INQUIRY` (or `RETURN_EXCHANGE_REQUEST`)
- **System Prediction:** `DELIVERY_SHIPPING_DELAY` (Confidence: 0.61)

### Why It Failed
The message contains tokens associated with three distinct intents:
1. "delayed by 2 weeks" $\rightarrow$ `DELIVERY_SHIPPING_DELAY`
2. "want to return it" $\rightarrow$ `RETURN_EXCHANGE_REQUEST`
3. "immediate refund on my card" $\rightarrow$ `REFUND_BILLING_INQUIRY`

The TF-IDF and semantic vectorizers detected high token frequency for "delayed" and "package", biasing the classifier toward the logistics class instead of the customer's ultimate actionable goal (monetary refund).

### Root Cause Hypothesis
Single-label classification cannot capture multi-intent hierarchical dependencies where a logistics failure causes a financial remedy request.

### Possible Fix
1. Implement multi-label intent classification with primary/secondary goal scoring.
2. Use an LLM with Chain-of-Thought prompting to identify the *actionable resolution* rather than the *triggering event*.

---

## 2. Failure Mode 2: Underspecified & Low-Information Queries

### Dataset Example
- **Customer Message:** *"My order is wrong."*
- **Gold Label:** `DAMAGED_DEFECTIVE_PRODUCT` (or `ORDER_CANCELLATION_MODIFICATION`)
- **System Prediction:** `OTHER_GENERAL_INQUIRY` (Confidence: 0.42)
- **Escalation Decision:** `ESCALATE` (Reason: Low intent confidence 0.42 < 0.65)

### Why It Failed
The query contains only 4 words (18 characters) with zero domain nouns (no mention of size, color, damaged item, missing item, or incorrect address).

### Root Cause Hypothesis
Under-specified customer inputs lack sufficient semantic density for cosine similarity or discriminative classification.

### How the System Handled It Correctly
Rather than hallucinating an arbitrary intent, our multi-signal escalation engine detected the low confidence ($0.42 < 0.65$) and triggered safe human escalation / clarification request, preventing incorrect automation.

### Possible Fix
Implement an interactive conversational clarification turn: *"Could you please clarify if you received a defective item, the wrong product, or an issue with your delivery address?"*

---

## 3. Failure Mode 3: Lexical Overlap vs. Semantic Policy Mismatch in Retrieval

### Dataset Example
- **Customer Message:** *"Can I request the courier to hold my package at an Amazon Locker instead of delivering to my house?"*
- **Retrieved Top-1 Evidence:** *"You can track your package directly under Your Orders or contact the carrier for delivery rescheduling."* (Similarity: 0.58)
- **Problem:** The retrieved evidence addresses standard home delivery tracking, failing to answer the specific policy question regarding **Locker redirection after dispatch**.

### Why It Failed
The words "courier", "package", "delivering", and "house" heavily overlap with standard tracking tweets in the vector index. The niche policy concept ("Amazon Locker in-flight rerouting") had lower lexical weight.

### Root Cause Hypothesis
Pure n-gram TF-IDF retrieval captures keyword co-occurrence but struggles with fine-grained constraint matching without dense transformer embeddings or structured metadata filtering.

### Possible Fix
1. Implement Hybrid Dense-Sparse Retrieval with Reciprocal Rank Fusion (RRF).
2. Add metadata filtering tagging conversations with specific sub-features (e.g., `feature:locker`, `feature:prime_video`).

---

## 4. Failure Mode 4: Over-Escalation on Casual Frustration Idioms

### Dataset Example
- **Customer Message:** *"This checkout bug is totally crazy and unacceptable, my coupon won't apply."*
- **Gold Action:** `AUTO_HANDLE` (Standard promo code FAQ)
- **System Action:** `ESCALATE`
- **Triggered Reason:** Matched keyword trigger `unacceptable`.

### Why It Failed
The escalation engine flagged the word "unacceptable", treating it as a severe customer service breakdown requiring human intervention, when in reality the core issue was a standard self-service promo code question.

### Root Cause Hypothesis
Keyword matching without syntactic or sentiment parsing treats mild colloquial venting identically to formal manager escalation requests.

### Possible Fix
1. Replace single-keyword triggers with context-aware sentiment analysis and dependency parsing.
2. Require both high frustration sentiment AND lack of self-service resolution before triggering sentiment-based escalation.

---

## 5. Failure Mode 5: Out-of-Distribution Enterprise & Digital Sub-Brand Queries

### Dataset Example
- **Customer Message:** *"Do you offer student discounts on AWS cloud certifications through Amazon retail?"*
- **Gold Label:** `OTHER_GENERAL_INQUIRY`
- **System Prediction:** `PROMO_DISCOUNT_PRICING` (Confidence: 0.52)

### Why It Failed
The message mentions "discounts", matching the `PROMO_DISCOUNT_PRICING` class, but belongs to AWS Educate rather than Amazon Retail customer support.

### Root Cause Hypothesis
The Twitter Customer Support dataset for `AmazonHelp` contains primarily retail e-commerce interactions. Digital B2B sub-brands (AWS, Twitch, Audible) are sparse in the retail training corpus.

### Possible Fix
1. Explicitly define negative boundary filters in `config/intents.yaml` for AWS and third-party subsidiaries.
2. Route out-of-scope enterprise queries to external departmental endpoints.
