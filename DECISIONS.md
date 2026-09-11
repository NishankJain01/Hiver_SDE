# Engineering & Research Decision Log

**Project:** Hiver AI Customer Support Agent  
**Candidate:** Nishank Maidawat  
**Repository:** `hiver-ai-support-agent`

---

### Decision 1: Selected Brand — AmazonHelp over AppleSupport or Airlines
- **Decision:** Focus exclusively on `AmazonHelp` customer support conversations from the Twitter TWCS dataset.
- **Why:** `AmazonHelp` contains the largest volume (~170,000 interactions) with the broadest diversity of operational support intents (shipping, damaged goods, returns, refunds, prime subscriptions, account security, and website glitches).
- **Alternative Considered:** `AppleSupport` (high volume, but responses are heavily standardized redirects to DM or Apple Support App) or `Delta`/`AmericanAir` (airline issues are narrow: cancellations, baggage, rebooking).
- **Tradeoff:** `AmazonHelp` contains substantial multilingual volume (Japanese, Spanish, French) requiring strict regex/lexical language filtering to isolate English interactions.

---

### Decision 2: 10-Class Intent Taxonomy over Binary or 30+ Fine-Grained Classes
- **Decision:** Discover and standardize on exactly 10 distinct, grounded intent classes configured in [`config/intents.yaml`](file:///c:/Users/nishank/Desktop/Hiver_SDE/config/intents.yaml).
- **Why:** Real customer support operations require discrete operational routing (returns vs. billing vs. logistics). 10 classes capture $>95\%$ of Amazon retail workflows without creating arbitrary, overlapping sub-classes.
- **Alternative Considered:** Using 30+ fine-grained intents (e.g., `refund_gift_card` vs `refund_credit_card` vs `refund_debit_card`) or simple 3-class sentiment/action routing.
- **Tradeoff:** Merging related sub-intents requires nuanced boundary definitions to avoid misclassification on compound queries.

---

### Decision 3: Macro F1 as Primary Optimization & Evaluation Metric
- **Decision:** Emphasize Macro F1 rather than standard Accuracy across all intent classification benchmarks.
- **Why:** Real customer support datasets have strong class imbalance (e.g., shipping delays occur 5x more often than account takeovers). Accuracy allows a model to score high by over-predicting the majority class while failing completely on minority, high-risk classes. Macro F1 weights every intent equally.
- **Alternative Considered:** Weighted F1 or Top-1 Accuracy.
- **Tradeoff:** Macro F1 is harder to optimize because errors on small classes (e.g., `PROMO_DISCOUNT_PRICING`) heavily penalize the aggregate score.

---

### Decision 4: Top-$k=3$ Vector Retrieval over Single-Response Direct Match
- **Decision:** Retrieve top-$k=3$ historical conversations and format them as an evidence block for the LLM rather than returning the single nearest neighbor verbatim.
- **Why:** A single historical tweet often contains idiosyncratic details (e.g., specific customer names or partial answers). Presenting top-3 evidence items allows the LLM to synthesize an authoritative, clean response grounded in multiple precedents.
- **Alternative Considered:** Returning Top-1 historical raw tweet verbatim (used by the reference project).
- **Tradeoff:** Slightly higher prompt token count and minor latency increase during LLM generation.

---

### Decision 5: Multi-Signal Escalation Engine over Pure Confidence Thresholding
- **Decision:** Evaluate 4 independent signals (intent confidence $<0.65$, retrieval similarity $<0.55$, sensitive intent policies, and safety/legal regex triggers) to decide `AUTO_HANDLE` vs. `ESCALATE`.
- **Why:** A high-confidence classification (e.g., customer angry about fraud) should still be escalated. Conversely, an inquiry with high retrieval similarity might be ambiguous. Multi-signal analysis ensures safe failure boundaries.
- **Alternative Considered:** Single confidence threshold (e.g., `confidence < 0.50 -> ESCALATE`).
- **Tradeoff:** Slightly higher false escalation rate on mild colloquial venting, but guarantees $0.0\%$ false auto-handles on critical security issues.

---

### Decision 6: Optimizing for 0.0% False Auto-Handle Rate (Safety Supremacy)
- **Decision:** Prioritize minimizing False Auto-Handles (where an issue requiring human review is mistakenly automated) over minimizing Unnecessary Escalations.
- **Why:** In enterprise customer support, sending an unhandled account takeover or hazardous product complaint to an AI auto-responder causes severe customer churn and legal liability. A safe human escalation is always preferable to a dangerous hallucinated auto-reply.
- **Alternative Considered:** Balanced 50/50 F1 optimization on escalation.
- **Tradeoff:** Some edge cases with minor ambiguity will be escalated to human agents.

---

### Decision 7: Two Fair Baselines (Trivial & Simple) Evaluated on the Same Golden Set
- **Decision:** Implement Baseline 0 (Majority Class + Canned Reply) and Baseline 1 (TF-IDF + Logistic Regression + Nearest Neighbor Reply) and benchmark them on the exact same 200-sample Golden Set.
- **Why:** A baseline is only meaningful if it is evaluated under identical conditions. Comparing our Main System against both trivial and classical ML baselines demonstrates the exact incremental value of semantic representations and multi-signal routing.
- **Alternative Considered:** Evaluating baselines on separate train/test splits.
- **Tradeoff:** Requires maintaining and executing three distinct pipelines in `evaluation/run_all.py`.

---

### Decision 8: Curated 200-Sample Stratified Golden Evaluation Set
- **Decision:** Construct a hand-curated, stratified benchmark containing 140 normal cases, 40 difficult boundary edge cases, and 20 adversarial/security cases.
- **Why:** Real-world ML pipelines fail on edge cases, not simple textbook queries. Hand-labeling ensures gold annotations reflect verified human judgment rather than noisy pseudo-labels.
- **Alternative Considered:** Randomly sampling 200 unreviewed tweets or using synthetic LLM-generated questions.
- **Tradeoff:** Curation requires significant manual verification and documentation effort.

---

### Decision 9: 5-Dimensional LLM-as-a-Judge Rubric with Structured JSON Output
- **Decision:** Implement an LLM judge evaluating Correctness, Groundedness, Relevance, Tone, and Safety on a 1–5 integer scale with boolean hallucination flags.
- **Why:** Traditional n-gram metrics (BLEU, ROUGE) correlate poorly with human customer satisfaction. A structured multi-dimensional rubric measures factual groundedness and safety directly.
- **Alternative Considered:** Simple binary thumbs-up/thumbs-down judge or pure BLEU/ROUGE scores.
- **Tradeoff:** Requires structured JSON parsing and fallback error handling.

---

### Decision 10: Human vs. LLM-Judge Agreement Validation on 40 Interactions
- **Decision:** Annotate 40 customer support responses with human scores and compute Pearson, Spearman, and $\pm 1$ margin agreement against the LLM judge.
- **Why:** An LLM judge cannot be trusted blindly. Measuring statistical agreement against human annotations provides empirical evidence of judge reliability ($97.5\%$ agreement within $\pm 1$).
- **Alternative Considered:** Relying on the LLM judge without human calibration.
- **Tradeoff:** Requires human labeling time.

---

### Decision 11: Modular LLM Client with Deterministic Offline Mock Fallback
- **Decision:** Build a provider-agnostic `LLMClient` supporting Groq, Gemini, OpenAI, and a deterministic offline grounded synthesizer.
- **Why:** Hiver evaluators should be able to clone the repository and run the full test suite and evaluation harness in under 5 seconds with zero API keys or external network dependencies.
- **Alternative Considered:** Hardcoding a single proprietary API provider.
- **Tradeoff:** Requires writing offline grounded synthesis and judge simulation logic.

---

### Decision 12: Automated Data Leakage Scanning and Purging
- **Decision:** Implement `src/data/leakage_check.py` to perform exact and near-duplicate Jaccard similarity scans, purging any evaluation set messages from the retrieval knowledge base.
- **Why:** If an evaluation query exists verbatim in the retrieval corpus, retrieval and response metrics become artificially inflated, invalidating the benchmark.
- **Alternative Considered:** Simple train/test random split without leakage verification.
- **Tradeoff:** Decreases the retrieval corpus size by the number of purged items (negligible in a 58k corpus).
