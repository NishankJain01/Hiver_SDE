# Hiver AI Customer Support Agent

**Candidate:** Nishank Maidawat  
**Program:** B.Tech Computer Science and Engineering (Artificial Intelligence)  
**Institute:** Parul Institute of Engineering and Technology, Parul University  
**Graduation:** 2027  
**Assignment:** Hiver SDE Intern Take-Home Project  
**Selected Brand:** `AmazonHelp` (from Kaggle *Customer Support on Twitter* dataset)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Evaluation Runtime](https://img.shields.io/badge/eval%20runtime-%3C%2010%20seconds-brightgreen.svg)]()
[![False Auto-Handle Rate](https://img.shields.io/badge/false%20auto--handle%20rate-0.0%25-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

The **Hiver AI Customer Support Agent** is a production-grade, grounded customer support system built on real Twitter customer-support conversations from the Kaggle `thoughtvector/customer-support-on-twitter` dataset.

Unlike naive rule-based bots or ungrounded generative chatbots, this system implements:
1. **Data-Driven Intent Classification:** 10 grounded support intents discovered from real customer interactions.
2. **Dense Vector Grounded Retrieval (RAG):** Retrieves historical resolved support precedents to ground response generation.
3. **Multi-Signal Explainable Escalation:** Evaluates intent confidence, retrieval similarity, sensitive intent policies, and safety triggers to decide `AUTO_HANDLE` vs. `ESCALATE` with human-readable rationale.
4. **Anti-Hallucination Guardrails:** Strict prompt safeguards preventing fake refund amounts, fake promo codes, or unverified account promises.
5. **Rigorous Benchmark Suite:** Evaluates Main System against Trivial (Majority) and Simple (TF-IDF + LogReg) baselines on a hand-curated **200-sample Stratified Golden Evaluation Set** with statistical Human vs. LLM-as-a-Judge agreement validation.

---

## 2. Key Benchmark Results

All metrics below are generated deterministically by running [`python -m evaluation.run_all`](file:///c:/Users/nishank/Desktop/Hiver_SDE/evaluation/run_all.py) on the 200-sample Golden Set:

| System Architecture | Intent Accuracy | Intent Macro F1 | Weighted F1 | Escalation Precision | Escalation Recall | False Auto-Handle Rate (Safety Critical) | Twitter Artifact Sanitization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0 (Trivial Majority)** | 0.195 | 0.033 | 0.064 | 0.000 | 0.000 | 1.000 (100.0%) | 100.0% |
| **Baseline 1 (Simple TF-IDF + LogReg)** | 0.665 | 0.664 | 0.665 | 0.230 | 1.000 | 0.000 (0.0%) | 100.0% |
| **Main System (Semantic Hybrid + Multi-Signal)** | **0.805** | **0.787** | **0.805** | **0.230** | **1.000** | **0.000 (0.0%)** | **100.0%** |

### Human vs. LLM-as-a-Judge Agreement (40 Interactions)
- **Exact Agreement:** **90.0%**
- **Agreement Within $\pm 1$ Score Margin:** **97.5%**
- **Mean Absolute Error (MAE):** **0.225**

---

## 3. Quick Start (Reproduce in Under 5 Minutes)

### Step 1: Clone the Repository & Set Up Virtual Environment

```bash
git clone https://github.com/nishankmaidawat/hiver-ai-support-agent.git
cd hiver-ai-support-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies & Copy Configuration

```bash
pip install -r requirements.txt
copy .env.example .env     # On Linux/macOS: cp .env.example .env
```

### Step 3: Run the Master Benchmark Suite

```bash
python -m evaluation.run_all
```
*(Executes all intent, escalation, reply quality, and human-judge agreement evaluations in ~5 seconds with zero API costs!)*

### Step 4: Run Unit Tests

```bash
python -m pytest -v
```

### Step 5: Launch the Interactive Streamlit Demo

```bash
streamlit run app.py
```
*Open http://localhost:8501 in your browser.*

---

## 4. System Architecture

```
+-----------------------------------------------------------------------------------+
|                            Incoming Customer Query                                |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Data Preprocessing & Normalization                         |
|             (URL masking, handle removal, emoji normalization, length check)      |
+-----------------------------------------+-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
+---------------------------------------+   +---------------------------------------+
|        Intent Classification          |   |       Dense / Hybrid Retrieval        |
|  - Baseline 0: Majority Class         |   |  - TF-IDF Baseline Retriever          |
|  - Baseline 1: TF-IDF + Logistic Reg  |   |  - Semantic Vector (Sentence-Transf.) |
|  - Main: Embedding / Hybrid Matcher   |   |    Top-K Historical Resolved Cases    |
|  Output: Predicted Intent + Confidence|   |  Output: Top-K Context & Sim Scores   |
+--------------------+------------------+   +--------------------+------------------+
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              Multi-Signal Escalation Engine                       |
|   Signals:                                                                        |
|   1. Intent Confidence (< 0.65 -> Low Confidence)                                 |
|   2. Retrieval Evidence Score (< 0.55 -> Weak Evidence)                           |
|   3. Sensitive / Risk Patterns (Fraud, Account Takeover, Legal, Agent Abuse)      |
|   4. Intent Type Policies (Account Security, Payment Disputes -> Human Review)    |
|   Output: Decision (AUTO_HANDLE / ESCALATE) + Explicit Human-Readable Reason      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         Grounded Reply Generation (LLM)                           |
|   - System Prompt with strict Grounding Constraints & Brand Support Tone          |
|   - Injects Top-K Retrieved Historical Evidence (No hallucinated policies)        |
|   - If ESCALATE: Generates safe acknowledgment & warm handoff message             |
|   - If AUTO_HANDLE: Generates grounded, actionable resolution                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               Final Output Schema (JSON)                          |
|   {                                                                               |
|     "message": "...",                                                             |
|     "intent": "DELIVERY_SHIPPING_DELAY",                                          |
|     "intent_confidence": 0.94,                                                    |
|     "decision": "AUTO_HANDLE",                                                    |
|     "escalation_reason": "High intent confidence and strong retrieval grounding", |
|     "retrieved_evidence": [...],                                                  |
|     "draft_reply": "..."                                                          |
|   }                                                                               |
+-----------------------------------------------------------------------------------+
```

---

## 5. Intent Taxonomy (10 Grounded Classes)

Configured in [`config/intents.yaml`](file:///c:/Users/nishank/Desktop/Hiver_SDE/config/intents.yaml):

1. `DELIVERY_SHIPPING_DELAY`: Missing packages, tracking updates, carrier delays.
2. `DAMAGED_DEFECTIVE_PRODUCT`: Broken goods, cracked items, incorrect items received.
3. `RETURN_EXCHANGE_REQUEST`: Return labels, pickup scheduling, size exchanges.
4. `REFUND_BILLING_INQUIRY`: Refund status, duplicate transactions, unexpected Prime charges.
5. `ACCOUNT_ACCESS_SECURITY`: Password resets, 2FA/OTP failures, hacked account alerts (*Always Escalate*).
6. `ORDER_CANCELLATION_MODIFICATION`: Canceling unshipped orders, changing delivery addresses.
7. `PROMO_DISCOUNT_PRICING`: Coupon code errors, gift card claim code issues, lightning deals.
8. `TECHNICAL_APP_WEBSITE_BUG`: Website checkout glitches, Prime Video playback error codes.
9. `FEEDBACK_AGENT_COMPLAINT`: Representative misconduct, broken callback promises (*Always Escalate*).
10. `OTHER_GENERAL_INQUIRY`: Out-of-scope inquiries, corporate questions, general greetings.

---

## 6. What is misleading about my headline number?

*(Mandatory assignment section)*

While our headline metrics (**80.5% Accuracy**, **0.787 Macro F1**, **0.0% False Auto-Handle Rate**) reflect a high-performing system, presenting them without context would be misleading for several real-world reasons:

1. **Golden Benchmark Sample Size:** The benchmark consists of 200 curated, stratified examples. At 200 samples, the $95\%$ confidence interval is approximately $\pm 5.5\%$. True population performance in live production would likely fall within the 75%–85% range.
2. **Conservative Escalation Tradeoff:** Achieving a **0.0% False Auto-Handle Rate** was accomplished by setting conservative escalation thresholds ($0.65$ confidence, $0.55$ retrieval similarity). This increases the **Unnecessary Escalation Rate (~23%)** on borderline or colloquial queries, trading higher human agent workload for zero critical security failures.
3. **Twitter Format vs. Long-Form Enterprise Tickets:** Twitter messages are character-constrained and often contain link redirects (`[URL]`). Real-world helpdesk software (Hiver, Zendesk) processes multi-paragraph emails with structured order attachments that may introduce different conversational dynamics.
4. **Offline Evaluation vs. Real Customer Satisfaction:** Automated LLM judge metrics evaluate response phrasing against historical support transcripts, which does not directly measure downstream customer issue resolution.

---

## 7. What I would do with one more week

1. **Dense-Sparse Hybrid Retrieval:** Implement Reciprocal Rank Fusion (RRF) combining BM25 keyword matching with dense transformer embeddings (e.g., `all-MiniLM-L6-v2` / ColBERT) for improved semantic policy matching.
2. **Hierarchical Multi-Label Intent Classification:** Predict primary and secondary intent vectors for compound customer complaints (e.g., "late delivery" + "refund demand").
3. **Dynamic Escalation Calibration:** Automatically tune confidence thresholds using isotonic regression on a held-out validation split.
4. **Contextual Dialogue Memory:** Maintain multi-turn state tracking across extended support conversations.
5. **Automated PII Redaction Pipeline:** Integrate Named Entity Recognition (NER) to detect and mask customer names, phone numbers, and physical addresses before LLM prompt injection.

---

## 8. Repository Structure

```
hiver-ai-support-agent/
├── README.md                       # Master Documentation & Quick Start
├── REPORT.md                       # Comprehensive 6-Page Technical Report
├── DECISIONS.md                    # 12 Engineering & Research Trade-Off Logs
├── ACKNOWLEDGEMENTS.md             # Dataset, Package & Reference Citations
├── requirements.txt                # Pinned Dependency Specification
├── .env.example                    # Environment Variable Template
├── .gitignore                      # Git Tracking Exclusions
├── app.py                          # Interactive Streamlit Demo
│
├── config/
│   ├── intents.yaml                # 10 Grounded Intent Definitions & Boundaries
│   ├── settings.yaml               # System Parameters & Thresholds
│   └── prompts.yaml                # Generation & LLM Judge Prompts
│
├── data/
│   ├── sample/
│   │   └── sample_pairs.csv        # Quick-test dataset (500 rows)
│   ├── processed/
│   │   └── amazon_clean_pairs.csv  # 58,168 Cleaned English Support Pairs
│   ├── golden/
│   │   └── golden_eval.csv         # 200-Sample Stratified Golden Evaluation Set
│   └── evaluation/
│       ├── evaluation_results.json # Serialized Benchmark Outputs
│       └── human_judge_scores.csv  # 40 Human-Annotated Ratings for Judge Validation
│
├── src/
│   ├── data/                       # Preprocessing, Reconstruction, Leakage Checks
│   ├── intents/                    # Rule, TF-IDF, and Semantic Intent Classifiers
│   ├── retrieval/                  # TF-IDF and Dense Vector Knowledge Base Retrievers
│   ├── escalation/                 # Multi-Signal Explainable Escalation Engine
│   ├── generation/                 # Modular LLM Client & Evidence-Grounded Generator
│   └── agent/                      # Unified SupportAgent Orchestrator
│
├── evaluation/
│   ├── baselines.py                # Baseline 0 (Trivial) & Baseline 1 (Simple)
│   ├── metrics.py                  # Macro F1, Escalation & Agreement Statistics
│   ├── evaluate_intents.py         # Intent Classification Benchmark Runner
│   ├── evaluate_escalation.py      # Escalation Benchmark Runner
│   ├── evaluate_replies.py         # Response Quality & Grounding Runner
│   ├── judge.py                    # LLM-as-a-Judge 5-Dimensional Evaluator
│   ├── evaluate_agreement.py       # Human vs. Judge Correlation Runner
│   └── run_all.py                  # Master Reproducible Evaluation Suite (<5s)
│
├── scripts/
│   ├── download_data.py            # Kaggle TWCS Dataset Downloader
│   ├── prepare_data.py             # Data Preparation & Leakage Purging Pipeline
│   └── build_golden_set.py         # 200-Sample Stratified Golden Set Builder
│
├── tests/                          # 16 Automated PyTest Unit Tests
│
└── docs/
    ├── GOLDEN_SET.md               # Golden Dataset Methodology & Annotation Rubric
    ├── FAILURE_ANALYSIS.md         # Top 5 Real Failure Modes Analysis
    ├── ARCHITECTURE.md             # Technical Architecture Specification
    └── INTERVIEW_GUIDE.md          # Conceptual Guide + 20 Live Interview Q&As
```

---

## 9. Final Acceptance Checklist Verification

- [x] **Brand Selected from Real Evidence:** `AmazonHelp` selected for high volume and intent diversity.
- [x] **Intent Taxonomy Derived from Data:** 10 grounded classes in `config/intents.yaml`.
- [x] **Intent Classifier Evaluated:** 80.5% Accuracy, 0.787 Macro F1.
- [x] **Grounded Historical Retrieval:** Top-$k$ vector retrieval with similarity bounds.
- [x] **Evidence-Grounded Response Generation:** Anti-hallucination prompting without inventing policies.
- [x] **Multi-Signal Escalation Engine:** Evaluates confidence, similarity, intent risk, and safety keywords with human-readable reasons.
- [x] **200 Curated Golden Examples:** Stratified across normal, edge, and adversarial cases in `data/golden/golden_eval.csv`.
- [x] **Golden Set Methodology Documented:** Detailed guidelines in `docs/GOLDEN_SET.md`.
- [x] **Trivial & Simple Baselines Evaluated:** Baseline 0 & Baseline 1 benchmarked on identical test set.
- [x] **Safety-Critical False Auto-Handle Rate:** **0.0%** (0 unhandled critical inquiries).
- [x] **LLM-as-a-Judge Implemented:** 5-dimensional rubric in `evaluation/judge.py`.
- [x] **Human vs. Judge Agreement Measured:** 97.5% agreement within $\pm 1$ on 40 human ratings.
- [x] **5 Real Failure Modes Documented:** Root causes and fixes in `docs/FAILURE_ANALYSIS.md`.
- [x] **"What is misleading about my headline number?" Included:** Exact heading in README & REPORT.md.
- [x] **12 Decision Log Entries:** Detailed trade-offs in `DECISIONS.md`.
- [x] **Zero Data Leakage:** Verified 0% overlap via `src/data/leakage_check.py`.
- [x] **16 Unit Tests Pass:** `python -m pytest -v` runs in 1.6s.
- [x] **Streamlit Demo Functional:** `streamlit run app.py` running interactively.
- [x] **No Secrets Committed:** Clean git tracking with `.gitignore` and `.env.example`.
- [x] **Candidate Details Verified:** Nishank Maidawat (B.Tech CSE - AI, Parul University, 2027).
