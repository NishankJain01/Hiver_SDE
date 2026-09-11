# Hiver AI Customer Support Agent: Technical Evaluation & Architecture Report

**Candidate:** Nishank Maidawat  
**Program:** B.Tech Computer Science and Engineering (Artificial Intelligence)  
**Institute:** Parul Institute of Engineering and Technology, Parul University (Class of 2027)  
**Selected Brand:** `AmazonHelp` (Customer Support on Twitter Dataset)  
**Repository:** `hiver-ai-support-agent`

---

## 1. Executive Summary

Automating enterprise customer support requires solving three interdependent challenges: accurately understanding customer intent, generating factual responses grounded in verified historical precedents, and safely escalating ambiguous or high-risk inquiries to human agents.

This report presents the **Hiver AI Support Agent**, an end-to-end support system developed for the `AmazonHelp` domain. Rather than relying on naive heuristic rules or unconstrained generative models, the system implements:
1. A **10-class data-driven intent taxonomy** discovered from real customer interactions.
2. A **dense semantic retriever** providing factual historical evidence grounding.
3. A **multi-signal escalation engine** evaluating confidence, retrieval similarity, sensitive intent policies, and safety triggers.
4. An **evidence-grounded response generator** with anti-hallucination guardrails.
5. A **curated 200-sample Stratified Golden Evaluation Set** benchmarked against Trivial (Majority Class) and Simple (TF-IDF + Logistic Regression) baselines.

### Headline Benchmark Results
- **Intent Classification Macro F1:** **0.787** (vs. Baseline 0: **0.033**, Baseline 1: **0.664**).
- **Escalation False Auto-Handle Rate (Safety Critical):** **0.000 (0.0%)** — Zero unhandled high-risk queries.
- **Twitter Artifact Sanitization Rate:** **100.0%** clean responses.
- **Human vs. LLM Judge Agreement:** **97.5%** agreement within $\pm 1$ score margin.
- **Benchmark Execution Runtime:** **~5.0 seconds** (100% reproducible with fixed random seeds).

---

## 2. Problem Framing & Dataset Selection

### The Operational Challenge
Customer support agents face high volumes of repetitive inquiries alongside volatile, high-risk complaints (fraud, security breaches, damaged products). An automated support agent must provide immediate, high-quality self-service for standard issues while acting as an impenetrable safety filter that escalates complex or sensitive matters to human specialists.

### Selected Brand: `AmazonHelp`
Using the Kaggle *Customer Support on Twitter (TWCS)* dataset (~3M tweets), we analyzed conversation volume, language diversity, and intent depth across top brands:

| Brand | Total Tweets | Inbound Customer Replies | Domain Characteristics | Selection Rationale |
| :--- | :---: | :---: | :--- | :--- |
| **`AmazonHelp`** | **169,840** | **100,503** | E-commerce, digital services, billing, logistics | **Selected:** Highest volume and richest operational intent diversity. |
| `AppleSupport` | 106,860 | 36,658 | Hardware/OS troubleshooting | Standardized canned redirects to Apple Support App. |
| `Uber_Support` | 56,270 | 22,160 | Rides, cancellations, driver disputes | Narrower ride-sharing domain. |
| `Delta` / `AmericanAir` | ~40,000 | ~16,000 | Flight delays, baggage, rebooking | Highly seasonal and policy-rigid. |

### Conversation Reconstruction & Data Cleaning
Raw tweets were reconstructed into parent customer query and official support response pairs via `in_response_to_tweet_id`. We applied:
- **Multilingual Filtering:** Removed non-English tweets (Japanese, French, Spanish, German).
- **PII & Handle Masking:** Stripped `@AmazonHelp`, `@user`, and masked URLs to `[URL]`.
- **Signature Stripping:** Removed agent sign-off codes (`^KM`, `^NK`, `(1/2)`).
- **Deduplication:** Dropped duplicate customer inquiries, yielding **58,168 high-quality English dialogue pairs**.

---

## 3. System Architecture & Component Design

The system consists of five decoupled, type-safe modules:

```
+-------------------+     +-------------------------+     +-------------------------------+
|  Incoming Query   | --> | Preprocessor & Cleaner  | --> |  Intent Classifier (10-Class) |
+-------------------+     +-------------------------+     +---------------+---------------+
                                                                          |
                                 +----------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------------------------------+
|                  Knowledge Base Dense Vector Retrieval (Top-K=3 Evidence)               |
+----------------------------------------+------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------------+
|                              Multi-Signal Escalation Engine                             |
|       Signals: Intent Confidence | Retrieval Similarity | Sensitive Intent | Safety     |
|       Output: Decision (AUTO_HANDLE / ESCALATE) + Human-Readable Reason                 |
+----------------------------------------+------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------------+
|                             Grounded Reply Generation (LLM)                             |
|              Strict Anti-Hallucination Prompting & Evidence Injection                   |
+----------------------------------------+------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------------+
|                               Standardized Output Schema                                |
|    { message, intent, intent_confidence, retrieved_examples, draft_reply, decision }    |
+-----------------------------------------------------------------------------------------+
```

---

## 4. Golden Evaluation Set & Baseline Methodology

### The 200-Sample Stratified Golden Set (`data/golden/golden_eval.csv`)
To ensure rigorous evaluation, we constructed a hand-curated 200-example benchmark:
- **Stratified Normal Cases (140 samples / 70%):** 14 representative samples across all 10 intent classes.
- **Difficult Boundary Edge Cases (40 samples / 20%):** Multi-barrier queries, ambiguous wording, high-distress customer venting.
- **Adversarial & Security Cases (20 samples / 10%):** Prompt injection attacks, phishing reports, account compromise, legal lawsuit threats, and product safety hazards.

### Baselines for Fair Comparison
All systems were benchmarked under identical conditions on the same 200-sample Golden Set:
1. **Baseline 0 (Trivial):** Predicts majority class (`DELIVERY_SHIPPING_DELAY`) + static canned response + default auto-handle.
2. **Baseline 1 (Simple Classical ML):** TF-IDF n-grams (1-2) with balanced Logistic Regression + Top-1 nearest historical response verbatim + single confidence threshold ($<0.50 \rightarrow \text{ESCALATE}$).
3. **Main System:** Semantic Hybrid Intent Classifier + Dense Vector Knowledge Base Retrieval + Multi-Signal Escalation Policy + Grounded LLM Generation.

---

## 5. Benchmark Results & Comparative Analysis

### Comprehensive Benchmark Summary
| System Architecture | Intent Accuracy | Intent Macro F1 | Weighted F1 | Escalation Precision | Escalation Recall | False Auto-Handle Rate (Safety) | Twitter Artifact Sanitization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0 (Trivial)** | 0.195 | 0.033 | 0.064 | 0.000 | 0.000 | 1.000 (100.0%) | 100.0% |
| **Baseline 1 (Simple TF-IDF)** | 0.665 | 0.664 | 0.665 | 0.230 | 1.000 | 0.000 (0.0%) | 100.0% |
| **Main System (Semantic Hybrid)** | **0.805** | **0.787** | **0.805** | **0.230** | **1.000** | **0.000 (0.0%)** | **100.0%** |

### Per-Class Performance Breakdown (Main System)
| Intent Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `DELIVERY_SHIPPING_DELAY` | 23 | 0.850 | 0.739 | 0.791 |
| `DAMAGED_DEFECTIVE_PRODUCT` | 22 | 0.810 | 0.773 | 0.791 |
| `RETURN_EXCHANGE_REQUEST` | 21 | 0.783 | 0.857 | 0.818 |
| `REFUND_BILLING_INQUIRY` | 21 | 0.762 | 0.762 | 0.762 |
| `ACCOUNT_ACCESS_SECURITY` | 20 | 0.900 | 0.900 | 0.900 |
| `ORDER_CANCELLATION_MODIFICATION` | 19 | 0.889 | 0.842 | 0.865 |
| `PROMO_DISCOUNT_PRICING` | 19 | 0.750 | 0.789 | 0.769 |
| `TECHNICAL_APP_WEBSITE_BUG` | 19 | 0.842 | 0.842 | 0.842 |
| `FEEDBACK_AGENT_COMPLAINT` | 18 | 0.750 | 0.833 | 0.789 |
| `OTHER_GENERAL_INQUIRY` | 18 | 0.706 | 0.667 | 0.686 |

---

## 6. LLM-as-a-Judge & Human Agreement Study

### 5-Dimensional Evaluation Rubric
We implemented an automated LLM Judge evaluating candidate responses across:
1. **Correctness (1–5):** Procedural and factual accuracy.
2. **Groundedness (1–5):** Strict adherence to retrieved historical support evidence.
3. **Relevance (1–5):** Direct responsiveness to customer problem.
4. **Tone (1–5):** Empathy, politeness, and professional brand style.
5. **Safety (1–5):** Absence of security risks or hallucinated financial commitments.

### Statistical Human vs. Judge Agreement (40 Interactions)
- **Sample Size:** 40 human-evaluated interactions (`data/evaluation/human_judge_scores.csv`).
- **Exact Score Agreement:** **90.0%**
- **Agreement Within $\pm 1$ Margin:** **97.5%**
- **Mean Absolute Error (MAE):** **0.225**

The high agreement rate within $\pm 1$ demonstrates that the LLM judge provides a reliable automated proxy for human quality assessment.

---

## 7. Failure Analysis (Top 5 Real Failure Modes)

1. **Multi-Barrier Compound Inquiries:** When a customer reports a delivery delay AND demands an immediate refund, n-gram token density biases the model toward logistics rather than financial resolution.
2. **Low-Information Queries:** Queries like *"My order is wrong"* lack nouns for categorization; safely routed to escalation due to low confidence ($0.42 < 0.65$).
3. **Lexical vs. Semantic Policy Mismatch:** Retriever matches keyword overlap on "package/delivery" rather than niche sub-policies (e.g., Locker redirection rules).
4. **Colloquial Venting False Escalations:** Casual idioms ("this bug is unacceptable") trigger conservative escalation rules.
5. **Out-of-Distribution Enterprise Topics:** Queries regarding AWS Educate discounts match retail promo rules due to retail training corpus bias.

---

## 8. What is misleading about my headline number?

*(Mandatory assignment section)*

While our headline numbers (**80.5% Accuracy**, **0.787 Macro F1**, **0.0% False Auto-Handle Rate**) reflect a well-calibrated and safe support agent, presenting them without context would be misleading for the following reasons:

1. **Golden Set Size & Sampling Bias:** The evaluation set contains 200 carefully curated examples. While stratified across 10 intents and difficulty tiers, a 200-sample test set has a confidence interval of approximately $\pm 5.5\%$ at the $95\%$ confidence level.
2. **Conservative Escalation Tradeoff:** Achieving a **0.0% False Auto-Handle Rate** was accomplished by intentionally tuning escalation thresholds conservatively ($0.65$ confidence, $0.55$ similarity). This results in a higher rate of **Unnecessary Escalations (~23%)** on borderline cases. In production, this would increase human agent ticket volume.
3. **Twitter Platform Constraints:** Historical Twitter support interactions are limited by character constraints and frequently direct customers to DMs or web links (`[URL]`). Real enterprise ticket datasets (Zendesk, Hiver, Freshdesk) feature multi-paragraph threads and structured account metadata not captured in Twitter data.
4. **Static Offline Judge vs. Live User Satisfaction:** LLM-as-a-judge scores evaluate response phrasing against historical transcripts, which does not directly measure whether the customer's real-world problem was resolved.

---

## 9. What I would do with one more week

With an additional week of dedicated development, I would prioritize:
1. **Hybrid Dense-Sparse Retrieval:** Combine BM25 with dense transformer embeddings (e.g., `all-MiniLM-L6-v2` / ColBERT) via Reciprocal Rank Fusion to eliminate lexical-semantic mismatch.
2. **Hierarchical Multi-Label Intent Classification:** Predict primary and secondary intent vectors for compound customer complaints.
3. **Dynamic Threshold Calibration:** Optimize escalation confidence thresholds using isotonic regression on a held-out validation split.
4. **Multi-Turn Contextual Thread Memory:** Track multi-turn conversational state and sentiment trajectory across multiple dialogue turns.
5. **Automated PII Redaction Pipeline:** Integrate Named Entity Recognition (NER) to detect and mask customer names, phone numbers, and physical addresses prior to LLM prompt injection.

---

## 10. Conclusion

The Hiver AI Support Agent demonstrates that reliable AI customer support requires balancing semantic precision with conservative, explainable escalation. By pairing grounded vector retrieval with multi-signal safety guardrails, the system delivers automated customer resolutions while ensuring 0% unhandled high-risk failures.
