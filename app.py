"""
Streamlit Web Application: Hiver AI Customer Support Agent.

Interactive prototype allowing evaluators to test intent classification,
evidence-grounded response generation, and multi-signal escalation.
"""

import os
import sys
import yaml
import pandas as pd
import streamlit as st

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agent.support_agent import SupportAgent
from src.data.reconstruct import load_or_process_pairs

# Streamlit Page Config
st.set_page_config(
    page_title="Hiver AI Support Agent | AmazonHelp",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_support_agent():
    """Initializes and caches the SupportAgent and knowledge base."""
    kb_path = "data/processed/amazon_clean_pairs.csv"
    if os.path.exists(kb_path):
        df_kb = pd.read_csv(kb_path, encoding="utf-8").head(10000)
    else:
        df_kb = load_or_process_pairs().head(5000)

    agent = SupportAgent(kb_df=df_kb)
    return agent, df_kb


@st.cache_data
def get_intents_taxonomy():
    """Loads intent taxonomy definitions."""
    with open("config/intents.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("intents", [])


def main():
    agent, df_kb = get_support_agent()
    intents_data = get_intents_taxonomy()

    # Header
    st.title("🤖 Hiver AI Customer Support Agent")
    st.caption("Grounded Customer Support Agent on Real Twitter Conversations | Brand: **AmazonHelp** | Candidate: **Nishank Maidawat**")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Agent Controls & Info")
        st.markdown("**Candidate:** Nishank Maidawat  \n**Institute:** Parul University  \n**Program:** B.Tech CSE (AI)")
        st.divider()

        st.subheader("📚 Knowledge Base Stats")
        st.metric(label="Indexed Historical Dialogues", value=f"{len(df_kb):,}")
        st.metric(label="Supported Support Intents", value=len(intents_data))

        st.divider()
        st.subheader("🔍 Explore Intent Taxonomy")
        selected_intent = st.selectbox(
            "Select an intent to inspect:",
            options=[item["id"] for item in intents_data],
            format_func=lambda x: next((f"{i['name']} ({i['id']})" for i in intents_data if i["id"] == x), x)
        )

        curr_intent_info = next((i for i in intents_data if i["id"] == selected_intent), None)
        if curr_intent_info:
            st.info(f"**Description:** {curr_intent_info['description']}")
            st.markdown(f"**Default Action:** `{curr_intent_info['default_action']}`")
            st.markdown("**Boundary Notes:**")
            st.caption(curr_intent_info.get("boundary_notes", "N/A"))

    # Main Interaction Area
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("📥 Incoming Customer Message")

        # Presets selector for quick evaluation
        preset_examples = {
            "Custom Query": "",
            "1. Delivery Delay": "My Prime package was supposed to arrive yesterday at 8 PM but tracking hasn't updated in 2 days. Where is it?",
            "2. Damaged Product": "The ceramic mug arrived completely shattered inside the box with broken glass everywhere.",
            "3. Return / Label Request": "How do I print a prepaid return shipping label for the shoes I received?",
            "4. Unexpected Refund / Charge": "Why was my credit card charged $14.99 for Prime when I canceled my membership last week?",
            "5. Account Security Alert (High Risk)": "Someone changed the email on my account and ordered $500 gift cards! I am locked out!",
            "6. Agent Misconduct Complaint (Escalation)": "I spoke to 4 different agents and none helped. Dave hung up on me. Connect me to a manager now.",
            "7. Legal Action Threat (Adversarial)": "My lawyer will be filing a formal lawsuit against Amazon in federal court on Monday."
        }

        chosen_preset = st.selectbox("⚡ Choose an evaluation scenario preset:", list(preset_examples.keys()))
        default_val = preset_examples[chosen_preset]

        query = st.text_area(
            "Enter customer inquiry text:",
            value=default_val,
            placeholder="Type a customer support message...",
            height=140
        )

        submit_btn = st.button("🚀 Process Inquiry with AI Agent", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Live Agent Execution Analysis")

        if submit_btn:
            if not query.strip():
                st.warning("Please enter a customer message to process.")
            else:
                with st.spinner("Analyzing message, classifying intent, retrieving evidence, and generating grounded reply..."):
                    result = agent.process_message(query)

                # 1. Decision & Status Banner
                decision = result["decision"]
                if decision == "AUTO_HANDLE":
                    st.success(f"### ✅ DECISION: {decision}")
                else:
                    st.error(f"### ⚠️ DECISION: {decision}")

                st.markdown(f"**Escalation Reason:**  \n*{result['escalation_reason']}*")

                st.divider()

                # 2. Intent Details
                c_intent, c_conf = st.columns([2, 1])
                with c_intent:
                    st.markdown(f"**Predicted Intent:**  \n`{result['intent']}`")
                with c_conf:
                    conf = result["intent_confidence"]
                    st.metric(label="Confidence Score", value=f"{conf * 100:.1f}%")
                    st.progress(conf)

                st.divider()

                # 3. Suggested Draft Response
                st.markdown("### 💬 Suggested Support Response")
                st.info(result["draft_reply"])

                # 4. Retrieved Historical Grounding Evidence
                st.markdown("### 📜 Retrieved Historical Support Evidence")
                evidence_list = result.get("retrieved_examples", [])
                if evidence_list:
                    for ev in evidence_list:
                        score = ev.get("similarity_score", 0.0)
                        cid = ev.get("conversation_id", ev.get("tweet_id", "N/A"))
                        with st.expander(f"Evidence #{ev.get('rank', 1)} | Similarity: {score:.2f} | ID: {cid}", expanded=(ev.get('rank', 1) == 1)):
                            st.markdown(f"**Historical Customer:** {ev.get('historical_customer_message', '')}")
                            st.markdown(f"**Amazon Response:** {ev.get('historical_support_response', ev.get('historical_response', ''))}")
                else:
                    st.caption("No historical evidence retrieved.")

                # 5. Raw Schema Inspector
                with st.expander("🛠️ View Raw JSON Response Schema"):
                    st.json(result)
        else:
            st.info("👈 Enter a customer inquiry or select a preset and click **Process Inquiry** to inspect live agent execution.")


if __name__ == "__main__":
    main()
