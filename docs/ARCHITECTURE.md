# System Architecture & Technical Specification

**Project:** Hiver AI Customer Support Agent  
**Candidate:** Nishank Maidawat (B.Tech CSE - AI, Parul University)  
**Selected Brand:** `AmazonHelp` (TWCS Dataset)

---

## 1. High-Level System Architecture

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

## 2. Core Subsystems

### 1. Data Cleaning & Conversation Reconstruction (`src/data/`)
- Reconstructs parent customer inquiries and authoritative brand responses using `in_response_to_tweet_id`.
- Strips Twitter relics: `@AmazonHelp`, `@115820`, `^KM`, `^NK`, `(1/2)`.
- Filters non-English messages using character ratio and lexical filtering.
- Implements leakage checks verifying 0% overlap with the golden evaluation set.

### 2. Intent Classification Engine (`src/intents/`)
- Discovered 10 data-grounded support intents configured in [`config/intents.yaml`](file:///c:/Users/nishank/Desktop/Hiver_SDE/config/intents.yaml).
- **Baseline 0:** Trivial majority class classifier.
- **Baseline 1:** TF-IDF n-gram (1-2) with balanced Logistic Regression.
- **Main System:** Semantic hybrid classifier combining class exemplar embeddings, centroid cosine similarity, and domain token boosting with temperature-calibrated softmax confidence.

### 3. Historical Knowledge Base Retrieval (`src/retrieval/`)
- Indexes 58,168 cleaned English AmazonHelp support dialogues.
- Returns top-$k$ evidence records with similarity scores.
- Formats evidence for LLM prompt injection and computes maximum grounding confidence bounds.

### 4. Multi-Signal Escalation Engine (`src/escalation/`)
- Replaces naive keyword matching with a multi-factor risk assessment:
  1. *Intent Confidence:* $< 0.65 \rightarrow$ Low confidence warning.
  2. *Retrieval Similarity:* $< 0.55 \rightarrow$ Insufficient grounding warning.
  3. *High-Risk Intent Types:* `ACCOUNT_ACCESS_SECURITY` and `FEEDBACK_AGENT_COMPLAINT` automatically route to human agents.
  4. *Safety & Legal Triggers:* Regex scan for fraud, theft, lawsuits, police, physical harm.
- Returns decision (`AUTO_HANDLE` vs. `ESCALATE`), explicit readable reason, and structured signal telemetry.

### 5. Grounded Response Generator (`src/generation/`)
- Injects customer message, predicted intent, escalation decision, and top retrieved historical evidence.
- Enforces strict anti-hallucination guardrails:
  - Never invent refund amounts or fake promo codes.
  - Never ask for passwords or credit card numbers.
  - Eliminate Twitter handles and agent sign-off codes.

---

## 3. Standardized Output Schema

The system produces a unified, type-safe JSON response:

```json
{
  "message": "My Prime package was supposed to arrive yesterday but tracking hasn't updated.",
  "cleaned_message": "My Prime package was supposed to arrive yesterday but tracking hasn't updated.",
  "intent": "DELIVERY_SHIPPING_DELAY",
  "intent_confidence": 0.945,
  "retrieved_examples": [
    {
      "rank": 1,
      "conversation_id": 100234,
      "similarity_score": 0.82,
      "historical_customer_message": "Where is my delayed package?",
      "historical_support_response": "You can track real-time shipping updates directly under Your Orders.",
      "is_sufficient_evidence": true
    }
  ],
  "draft_reply": "We apologize for the delay. You can track real-time carrier updates directly under 'Your Orders' in your Amazon account.",
  "decision": "AUTO_HANDLE",
  "escalation_reason": "High intent confidence (0.95) and strong historical evidence grounding (similarity 0.82). Safe to auto-handle.",
  "risk_signals": {
    "low_intent_confidence": false,
    "weak_retrieval_evidence": false,
    "high_risk_intent": false,
    "critical_keyword_trigger": false,
    "ambiguous_or_empty": false,
    "matched_triggers": []
  }
}
```
