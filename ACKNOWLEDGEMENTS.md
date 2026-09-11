# Acknowledgements & Citations

**Candidate:** Nishank Maidawat  
**Program:** B.Tech Computer Science and Engineering (Artificial Intelligence)  
**Institute:** Parul Institute of Engineering and Technology, Parul University  
**Assignment:** Hiver SDE Intern Take-Home Project

---

## 1. Datasets & Primary Sources

- **Primary Dataset:** *Customer Support on Twitter (TWCS)*  
  - **Dataset Identifier:** `thoughtvector/customer-support-on-twitter` (Kaggle)  
  - **Description:** Multi-turn customer-support interactions between users and global corporate brands on Twitter.  
  - **Usage:** Used as the underlying conversational source to extract and pair 58,168 clean English support dialogues for the `AmazonHelp` brand.

---

## 2. Reference Repository Attribution

In compliance with Hiver's assignment guidelines regarding transparency and citation of external inspirations:

- **Reference Repository:** [`jhanvimodani/hiver-support-agent`](https://github.com/jhanvimodani/hiver-support-agent)  
- **Nature of Attribution:** **Referenced for High-Level Domain Context**  
- **What Was Referenced / Inspired:**  
  1. The selection of `AmazonHelp` as a high-volume brand within the Kaggle TWCS dataset.  
  2. The concept of utilizing `in_response_to_tweet_id` to establish parent-child relationships between customer tweets and brand replies.  
- **What Is Original & Independent to This Repository:**  
  1. Complete modular architecture (`src/data/`, `src/intents/`, `src/retrieval/`, `src/escalation/`, `src/generation/`, `src/agent/`).  
  2. Rigorous data preprocessor with multilingual filtering, Twitter handle masking, URL normalization, and agent signature stripping.  
  3. Formal 10-class grounded intent taxonomy with boundary definitions and few-shot exemplars (`config/intents.yaml`).  
  4. Two fair baselines (Trivial Majority and Simple TF-IDF + Logistic Regression) evaluated on the exact same benchmark.  
  5. Dense vector semantic retriever with similarity score bounds.  
  6. Multi-signal explainable escalation engine (evaluating confidence, retrieval similarity, sensitive intent policies, and safety keywords).  
  7. Evidence-grounded LLM response generation with strict anti-hallucination prompting.  
  8. Curated 200-sample Stratified Golden Evaluation Set (`data/golden/golden_eval.csv`) with full manual annotations and leakage verification.  
  9. Multidimensional LLM-as-a-Judge evaluation suite with human agreement correlation validation.  
  10. Full automated test suite (`pytest`) and reproducible `<15 min` evaluation runner (`python -m evaluation.run_all`).

---

## 3. Open-Source Libraries & Frameworks

We gratefully acknowledge the developers of the following open-source packages:

- **Scikit-Learn:** Machine learning pipelines, TF-IDF vectorization, Logistic Regression, and classification metrics.
- **Pandas & NumPy:** Dataframe manipulation, fast vector operations, and statistical calculations.
- **SciPy:** Pearson and Spearman rank correlation coefficient calculations.
- **Streamlit:** Interactive web application framework.
- **PyTest:** Automated unit and regression test suite.
- **PyYAML & Pydantic:** Schema validation and configuration management.
- **Requests & HTTPX:** HTTP client interactions.
