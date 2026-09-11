# Golden Evaluation Set Methodology & Annotation Guide

**Project:** Hiver AI Customer Support Agent  
**Dataset File:** [`data/golden/golden_eval.csv`](file:///c:/Users/nishank/Desktop/Hiver_SDE/data/golden/golden_eval.csv)  
**Total Curated Examples:** 200  
**Curator & Lead Annotator:** Nishank Maidawat (B.Tech CSE - AI, Parul University)

---

## 1. Overview and Objective

The primary principle of this project is that **"The proof is worth more than the system."** Evaluating customer support systems on arbitrary synthetic data or circular self-generated pseudo-labels produces misleadingly high metrics.

To establish authentic credibility, we constructed a **200-sample Stratified Golden Evaluation Set** drawn from real Twitter customer inquiries and curated edge cases. Every single example was manually reviewed, labeled, and validated.

---

## 2. Dataset Schema

Each row in `data/golden/golden_eval.csv` contains the following structured fields:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `example_id` | Integer | Unique identifier (1 to 200). |
| `tweet_id` | Integer | Original Twitter parent tweet ID or synthetic benchmark ID. |
| `customer_message` | String | Cleaned, normalized customer inquiry text. |
| `conversation_context` | String | Contextual information (e.g., order status, multi-turn history). |
| `gold_intent` | String | Ground-truth intent class from the 10 data-driven taxonomy classes. |
| `gold_action` | String | Ground-truth decision: `AUTO_HANDLE` vs. `ESCALATE`. |
| `gold_escalation_reason` | String | Explicit human-interpretable rationale for the decision. |
| `human_reference_reply` | String | High-quality human support response serving as gold reference. |
| `difficulty` | String | Category: `NORMAL`, `EDGE_CASE`, or `ADVERSARIAL`. |
| `sampling_bucket` | String | `random_stratified`, `difficult_edge_case`, or `adversarial_security`. |
| `annotator` | String | Name of the human annotator validating the label. |

---

## 3. Sampling Strategy and Class Coverage

The dataset is stratified across all 10 customer support intents to prevent majority-class bias from obscuring critical errors in low-frequency categories.

### Distribution Breakdown
- **Normal Inquiries (140 examples / 70%):** Standard customer support requests with clear intent signals and representative phrasing.
- **Difficult Edge Cases (40 examples / 20%):** Complex multi-issue inquiries, high customer frustration, ambiguous phrasing, or boundary disputes between overlapping classes (e.g., damaged product vs. return vs. refund).
- **Adversarial & Security Cases (20 examples / 10%):** Prompt injection attacks, phishing reports, account compromise alerts, legal lawsuit threats, and hazardous product complaints.

### Intent Coverage Table
| Intent ID | Intent Name | Normal | Edge | Adversarial | Total | Default Action |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `DELIVERY_SHIPPING_DELAY` | Delivery & Shipping Delay | 14 | 4 | 2 | 20 | AUTO_HANDLE |
| `DAMAGED_DEFECTIVE_PRODUCT` | Damaged or Defective Product | 14 | 4 | 2 | 20 | AUTO_HANDLE |
| `RETURN_EXCHANGE_REQUEST` | Return & Exchange Request | 15 | 4 | 1 | 20 | AUTO_HANDLE |
| `REFUND_BILLING_INQUIRY` | Refund & Billing Inquiry | 14 | 4 | 2 | 20 | AUTO_HANDLE |
| `ACCOUNT_ACCESS_SECURITY` | Account Access & Security | 12 | 5 | 3 | 20 | ESCALATE |
| `ORDER_CANCELLATION_MODIFICATION` | Order Cancel / Modify | 15 | 4 | 1 | 20 | AUTO_HANDLE |
| `PROMO_DISCOUNT_PRICING` | Promo & Pricing Inquiry | 15 | 4 | 1 | 20 | AUTO_HANDLE |
| `TECHNICAL_APP_WEBSITE_BUG` | Tech & App Glitches | 15 | 4 | 1 | 20 | AUTO_HANDLE |
| `FEEDBACK_AGENT_COMPLAINT` | Agent Complaint / Frustration | 10 | 8 | 2 | 20 | ESCALATE |
| `OTHER_GENERAL_INQUIRY` | Other & General Inquiry | 12 | 3 | 5 | 20 | AUTO_HANDLE |
| **TOTAL** | | **136** | **44** | **20** | **200** | |

---

## 4. Annotation Guidelines & Decision Rubric

### A. Intent Classification Guidelines
1. **Primary Intent Rule:** When a message contains multiple mentions (e.g., "My package arrived late AND the item inside is broken"), assign the intent based on the **core unresolved barrier**. In the example, the damage is the permanent issue $\rightarrow$ `DAMAGED_DEFECTIVE_PRODUCT`.
2. **Financial vs. Logistics:** If the customer asks "Where is my refund for the returned jacket?", classify under `REFUND_BILLING_INQUIRY`. If they ask "How do I ship the jacket back?", classify under `RETURN_EXCHANGE_REQUEST`.
3. **Security Supremacy:** Any mention of unauthorized logins, OTP bypass, or stolen accounts MUST be classified as `ACCOUNT_ACCESS_SECURITY` regardless of secondary delivery complaints.

### B. Action Decision Guidelines (`AUTO_HANDLE` vs. `ESCALATE`)
An inquiry is labeled **`ESCALATE`** if ANY of the following apply:
1. **Security & Authentication:** Account takeover, password reset without 2FA, unauthorized charges.
2. **Safety & Hazard:** Electrical sparks, battery swelling, fire, or physical injury.
3. **Legal & Threat:** Mention of lawsuit, attorney, police, consumer protection bureau, or fraud charges.
4. **Repetitive Failure & Agent Misconduct:** Multiple broken callback promises, rude representative behavior, or manager escalation demands.
5. **Policy Exceptions:** Requests to bypass return windows or override warranty terms.

Otherwise, if standard self-service paths exist (e.g., Your Orders tracking, Returns Center QR label, standard refund buffers), the label is **`AUTO_HANDLE`**.

---

## 5. Quality Control and Leakage Prevention

1. **Exact & Near-Duplicate Deduplication:** Before evaluation, the script [`src/data/leakage_check.py`](file:///c:/Users/nishank/Desktop/Hiver_SDE/src/data/leakage_check.py) scans the retrieval Knowledge Base against `golden_eval.csv` to ensure 0% lexical or semantic overlap.
2. **Deterministic Reproducibility:** Fixed random seeds (`seed=42`) ensure all samples and splits remain perfectly identical across test runs.
