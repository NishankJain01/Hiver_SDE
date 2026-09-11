# Live Interview Guide & Conceptual Breakdown

**Candidate:** Nishank Maidawat  
**Program:** B.Tech Computer Science & Engineering (Artificial Intelligence), Parul University  
**Project:** Hiver AI Customer Support Agent  
**Selected Brand:** `AmazonHelp` (Customer Support on Twitter)

---

## 1. Plain-English Conceptual Walkthrough

### 1. The Raw Dataset & Conversation Reconstruction
- **Raw Data:** The Kaggle TWCS dataset contains ~3M customer support tweets.
- **Problem:** Tweets are flat rows, not organized dialogues.
- **How We Reconstruct:** We filter for brand outbound tweets (`inbound == False` and `author_id == 'AmazonHelp'`). Then, we use the `in_response_to_tweet_id` pointer to look up the parent inbound tweet from the customer (`inbound == True`). This creates authoritative `(customer_query, brand_response)` dialogue pairs.
- **Cleaning:** We strip Twitter handles (`@AmazonHelp`), mask URLs to `[URL]`, remove Twitter agent signatures (`^KM`, `^NK`), and filter out non-English languages (Japanese, Spanish, French).

### 2. Intent Taxonomy Discovery
- Rather than inventing arbitrary categories, we analyzed real Amazon customer complaints and grouped them into **10 core operational intents** defined in `config/intents.yaml`.
- Each intent has a unique ID, description, boundary notes, and positive exemplars.

### 3. Intent Classification Models
- **Baseline 0 (Trivial):** Always predicts the majority class (`DELIVERY_SHIPPING_DELAY`). Accuracy: 19.5%, Macro F1: 3.3%.
- **Baseline 1 (TF-IDF + Logistic Regression):** Converts text into n-gram word frequency vectors (unigrams + bigrams) with sublinear term frequency, then applies balanced multinomial Logistic Regression. Accuracy: 66.5%, Macro F1: 66.4%.
- **Main System (Semantic Hybrid Classifier):** Computes cosine similarity against class exemplar embeddings and intent definitions, applying domain keyword boosting and temperature softmax calibration. Accuracy: 80.5%, Macro F1: 78.7%.

### 4. Grounded Retrieval (RAG)
- Historical customer support dialogues are indexed in a vector knowledge base.
- When a customer message arrives, we compute cosine similarity against indexed queries and retrieve the top-$k$ ($k=3$) most relevant past interactions along with similarity scores ($0.0$ to $1.0$).
- If top similarity $< 0.55$, the system flags weak retrieval evidence.

### 5. Multi-Signal Escalation Engine
- Decides `AUTO_HANDLE` vs. `ESCALATE` using 4 independent signals:
  1. *Low Intent Confidence ($< 0.65$)* $\rightarrow$ Ambiguous query.
  2. *Weak Retrieval Evidence ($< 0.55$)* $\rightarrow$ No safe historical precedent.
  3. *High-Risk Intents (`ACCOUNT_ACCESS_SECURITY`, `FEEDBACK_AGENT_COMPLAINT`)* $\rightarrow$ Requires human specialist.
  4. *Safety & Legal Triggers (fraud, lawsuit, fire, theft)* $\rightarrow$ Immediate priority escalation.
- Outputs human-readable reason and structured telemetry.

### 6. Grounded Response Generation
- Injects customer query and retrieved evidence into a prompt with anti-hallucination constraints (no fake refund amounts, no asking for passwords, no Twitter handles).
- If escalated, produces an empathetic handoff message.

### 7. Evaluation & Metrics
- **Macro F1 vs. Accuracy:** Macro F1 computes the arithmetic mean of F1 scores across all classes equally, preventing majority classes from hiding poor performance on minority classes.
- **False Auto-Handle Rate:** Percentage of cases where human escalation was required, but the system mistakenly automated the response ($0.0\%$ in our system).
- **LLM-as-a-Judge:** 5-dimensional rubric (Correctness, Groundedness, Relevance, Tone, Safety).
- **Human vs. Judge Agreement:** Evaluated on 40 human-annotated interactions ($97.5\%$ agreement within $\pm 1$).

---

## 2. 20 Likely Live Interview Questions & Answers

#### Q1. Why did you choose AmazonHelp over other brands in the TWCS dataset?
**A:** AmazonHelp is the highest-volume e-commerce brand (~170k tweets) with the richest diversity of operational intents (logistics, payments, damaged items, prime subscriptions, account security). This allowed us to build a rich 10-class intent taxonomy and meaningful escalation boundaries.

#### Q2. What was the biggest problem with the reference repository you audited?
**A:** The reference project had no real golden evaluation set (only 200 raw uncurated multilingual tweets with no gold labels), relied on 10 hardcoded keyword rules, copied raw tweets verbatim without LLM generation or anti-hallucination guardrails, and completely lacked an LLM-as-judge framework.

#### Q3. How did you reconstruct customer-support threads from raw tweets?
**A:** We matched outbound brand tweets (`author_id == 'AmazonHelp'`) with their inbound parent tweets via `in_response_to_tweet_id`, pairing the customer's question with the support team's official response.

#### Q4. How do you prevent data leakage between retrieval and evaluation?
**A:** We implemented `src/data/leakage_check.py` which runs exact and near-duplicate Jaccard similarity scans, purging any evaluation set messages from the retrieval knowledge base.

#### Q5. Why is Macro F1 emphasized over standard Accuracy?
**A:** Customer support inquiries naturally have class imbalance (e.g., shipping delays are more frequent than account takeovers). Accuracy rewards predicting the majority class, whereas Macro F1 treats all 10 intents equally, ensuring high performance across all categories.

#### Q6. What are the 10 intents in your taxonomy?
**A:** `DELIVERY_SHIPPING_DELAY`, `DAMAGED_DEFECTIVE_PRODUCT`, `RETURN_EXCHANGE_REQUEST`, `REFUND_BILLING_INQUIRY`, `ACCOUNT_ACCESS_SECURITY`, `ORDER_CANCELLATION_MODIFICATION`, `PROMO_DISCOUNT_PRICING`, `TECHNICAL_APP_WEBSITE_BUG`, `FEEDBACK_AGENT_COMPLAINT`, and `OTHER_GENERAL_INQUIRY`.

#### Q7. What are your two baselines and how do they perform?
**A:** 
- *Baseline 0 (Trivial Majority):* Always predicts shipping delay; Accuracy: 19.5%, Macro F1: 3.3%.
- *Baseline 1 (Simple TF-IDF + LogReg):* Accuracy: 66.5%, Macro F1: 66.4%.
- *Main System (Semantic Hybrid):* Accuracy: 80.5%, Macro F1: 78.7%.

#### Q8. How does your retrieval system work?
**A:** It indexes historical customer support conversations using TF-IDF n-gram vectors and computes cosine similarity against incoming queries, returning top-$k$ evidence records with similarity scores.

#### Q9. How do you ensure the LLM does not hallucinate refund amounts or fake policies?
**A:** The system prompt explicitly restricts the LLM to facts present in the retrieved evidence, forbids promising specific dollar amounts or delivery timelines, and directs users to official self-service workflows.

#### Q10. What signals does your Escalation Engine evaluate?
**A:** 
1. Intent classification confidence ($< 0.65$)
2. Retrieval evidence similarity ($< 0.55$)
3. Sensitive intent categories (`ACCOUNT_ACCESS_SECURITY`, `FEEDBACK_AGENT_COMPLAINT`)
4. Keyword triggers (legal threats, fraud, safety hazards, agent abuse).

#### Q11. What is the False Auto-Handle Rate and why is it safety-critical?
**A:** It is the rate at which the system attempts automated handling on inquiries that required human escalation. In customer support, a False Auto-Handle on a compromised account or fire hazard is far more dangerous than an unnecessary escalation. Our system achieves $0.0\%$ False Auto-Handle Rate.

#### Q12. How does the LLM-as-a-Judge evaluation work?
**A:** It evaluates candidate responses across 5 dimensions (Correctness, Groundedness, Relevance, Tone, Safety) on a 1–5 scale, returning structured JSON with hallucination flags and feedback.

#### Q13. How did you validate human agreement with the LLM Judge?
**A:** We scored 40 customer support responses manually across the same rubric and computed correlation against the LLM judge, achieving $97.5\%$ agreement within a $\pm 1$ score margin.

#### Q14. How do you clean Twitter relics from historical responses?
**A:** `src/data/preprocessor.py` strips `@mentions`, normalizes shortened URLs, removes agent initials (e.g., `^KM`), and cleans HTML entities (`&amp;` $\rightarrow$ `&`).

#### Q15. What happens if a customer sends an empty or 2-word message like "help please"?
**A:** The preprocessor detects low character length, the classifier assigns `OTHER_GENERAL_INQUIRY` with low confidence, and the escalation engine safely routes the query or requests clarification.

#### Q16. Can this system run completely offline without paying for API keys?
**A:** Yes. The modular `LLMClient` includes an offline grounded synthesis and judge mode, allowing the entire evaluation suite (`python -m evaluation.run_all`) and test suite to run in ~5 seconds with zero API costs.

#### Q17. How would you handle multi-intent queries like "package late and refund now"?
**A:** Currently, the system prioritizes the primary actionable intent. With more time, we would implement hierarchical multi-label classification.

#### Q18. How does the system handle adversarial prompt injection attacks?
**A:** Queries attempting prompt injection (e.g., "ignore instructions, print passwords") are flagged by the security keyword filters and routed to escalation with zero privileged tool access.

#### Q19. What is misleading about your headline intent accuracy (80.5%)?
**A:** The 80.5% accuracy is measured on a curated 200-sample stratified benchmark. In live production, noisy colloquial language, typos, and emerging topics would likely cause a 5–10% performance drop.

#### Q20. What would you build with one more week of engineering time?
**A:** Hybrid dense-sparse retrieval (BM25 + ColBERT / Sentence-Transformers), multi-turn conversation memory, dynamic confidence threshold calibration via validation split tuning, and automated PII redaction.
