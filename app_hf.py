"""
Gradio Web App: Hiver AI Customer Support Agent.
Deployed on Hugging Face Spaces (free tier via Gradio SDK).

Provides the same functionality as app.py but using Gradio for HF compatibility.
"""

import os
import sys
import yaml
import pandas as pd
import gradio as gr

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agent.support_agent import SupportAgent
from src.data.reconstruct import load_or_process_pairs

# ── Load once ──────────────────────────────────────────────────────────────────
KB_PATH = "data/processed/amazon_clean_pairs.csv"

def _load_agent():
    if os.path.exists(KB_PATH):
        df_kb = pd.read_csv(KB_PATH, encoding="utf-8").head(8000)
    else:
        df_kb = load_or_process_pairs().head(5000)
    agent = SupportAgent(kb_df=df_kb)
    return agent, df_kb

agent, df_kb = _load_agent()

with open("config/intents.yaml", "r", encoding="utf-8") as f:
    intents_data = yaml.safe_load(f).get("intents", [])

INTENT_NAMES = {i["id"]: i["name"] for i in intents_data}

# ── Preset examples ────────────────────────────────────────────────────────────
PRESETS = {
    "1. Delivery Delay": "My Prime package was supposed to arrive yesterday at 8 PM but tracking hasn't updated in 2 days. Where is it?",
    "2. Damaged Product": "The ceramic mug arrived completely shattered inside the box with broken glass everywhere.",
    "3. Return / Label Request": "How do I print a prepaid return shipping label for the shoes I received?",
    "4. Unexpected Charge": "Why was my credit card charged $14.99 for Prime when I canceled my membership last week?",
    "5. Account Security (High Risk)": "Someone changed the email on my account and ordered $500 gift cards! I am locked out!",
    "6. Agent Misconduct (Escalation)": "I spoke to 4 different agents and none helped. Dave hung up on me. Connect me to a manager now.",
    "7. Legal Threat (Adversarial)": "My lawyer will be filing a formal lawsuit against Amazon in federal court on Monday.",
}

# ── Core inference function ────────────────────────────────────────────────────
def process_query(message: str):
    if not message or not message.strip():
        return (
            "⚠️ Please enter a customer message.",
            "", "", "", "", ""
        )

    result = agent.process_message(message)

    decision = result["decision"]
    decision_icon = "✅ AUTO_HANDLE" if decision == "AUTO_HANDLE" else "⚠️ ESCALATE"

    intent_id = result["intent"]
    intent_name = INTENT_NAMES.get(intent_id, intent_id)
    conf = result["intent_confidence"]
    intent_str = f"**{intent_name}** (`{intent_id}`)  \nConfidence: **{conf*100:.1f}%**"

    escalation = result["escalation_reason"]

    reply = result["draft_reply"]

    evidence_list = result.get("retrieved_examples", [])
    ev_lines = []
    for ev in evidence_list:
        score = ev.get("similarity_score", 0.0)
        cid   = ev.get("conversation_id", "N/A")
        cust  = ev.get("historical_customer_message", "")
        resp  = ev.get("historical_support_response", "")
        ev_lines.append(
            f"**Evidence #{ev.get('rank',1)}** | Similarity: {score:.2f} | ID: {cid}\n"
            f"> 👤 *Customer:* {cust}\n"
            f"> 🏢 *Amazon:* {resp}"
        )
    evidence_md = "\n\n---\n\n".join(ev_lines) if ev_lines else "*No historical evidence retrieved.*"

    stats = (
        f"**Knowledge Base:** {len(df_kb):,} indexed dialogues  \n"
        f"**Intents supported:** {len(intents_data)}  \n"
        f"**LLM Provider:** Mock (offline, deterministic)"
    )

    return decision_icon, intent_str, escalation, reply, evidence_md, stats


def load_preset(preset_name: str):
    return PRESETS.get(preset_name, "")


# ── Gradio UI ──────────────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; }
.gr-button-primary { background: linear-gradient(135deg, #7C3AED, #4F46E5) !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Hiver AI Customer Support Agent",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
    ),
    css=CSS
) as demo:

    gr.HTML("""
    <div id="title">
      <h1>🤖 Hiver AI Customer Support Agent</h1>
      <p style="color:#6B7280">
        Grounded Customer Support Agent on Real Twitter Conversations &nbsp;|&nbsp;
        Brand: <strong>AmazonHelp</strong> &nbsp;|&nbsp;
        Candidate: <strong>Nishank Maidawat</strong> &nbsp;|&nbsp;
        Hiver SDE Intern Assignment
      </p>
    </div>
    """)

    with gr.Row():
        # ── Left column: Input ─────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Customer Inquiry")

            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                label="⚡ Quick Evaluation Presets",
                value=None,
                interactive=True
            )
            query_box = gr.Textbox(
                label="Customer Message",
                placeholder="Type a customer support message here...",
                lines=5
            )
            submit_btn = gr.Button("🚀 Process with AI Agent", variant="primary", size="lg")

            gr.Markdown("### 📊 System Info")
            stats_out = gr.Markdown(
                f"**Knowledge Base:** {len(df_kb):,} indexed dialogues  \n"
                f"**Intents supported:** {len(intents_data)}  \n"
                f"**LLM Provider:** Mock (offline, deterministic)"
            )

            gr.Markdown("### 📚 Intent Taxonomy")
            intent_list_md = "\n".join(
                f"- **{i['name']}** — {i['description']}" for i in intents_data
            )
            gr.Markdown(intent_list_md)

        # ── Right column: Output ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Agent Decision")
            decision_out = gr.Markdown("*Submit a query to see the agent decision.*")

            gr.Markdown("### 🏷️ Intent Classification")
            intent_out = gr.Markdown()

            gr.Markdown("### 📋 Escalation Reason")
            escalation_out = gr.Markdown()

            gr.Markdown("### 💬 Suggested Support Reply")
            reply_out = gr.Textbox(label="Draft Reply", lines=4, interactive=False)

            gr.Markdown("### 📜 Retrieved Historical Evidence")
            evidence_out = gr.Markdown()

    # ── Event wiring ───────────────────────────────────────────────────────────
    preset_dd.change(fn=load_preset, inputs=preset_dd, outputs=query_box)

    submit_btn.click(
        fn=process_query,
        inputs=query_box,
        outputs=[decision_out, intent_out, escalation_out, reply_out, evidence_out, stats_out]
    )

    gr.Markdown("""
    ---
    **Built by Nishank Maidawat** for the Hiver SDE Intern Take-Home Assignment.
    Source code: [github.com/NishankJain01/Hiver_SDE](https://github.com/NishankJain01/Hiver_SDE)
    """)


if __name__ == "__main__":
    demo.launch()
