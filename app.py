"""SWE015 - Customer Support Ticket Router (Gradio app).

Run:  python app.py            (local, http://127.0.0.1:7860)
      python app.py --share    (public link, e.g. on Google Colab)
"""
import argparse

import gradio as gr
import pandas as pd

from scripts.download_models import ensure_models
from src.router import Router, DISPLAY, PRIMARY

ensure_models()
router = Router()

EXAMPLES = [
    "bulaşık makinem su boşaltmıyor, model DW-6534",
    "my dishwaser is leaking water everywhere",
    "I lost my credit card, please block it",
    "can you book a table for two at an italian restaurant tonight",
    "Hellooo! how are you?",
    "what is the best pizza topping on mars",
    "the fridge model RF-5500 stopped cooling after the power cut",
]


def handle(text):
    if not text or not text.strip():
        return "Please type a message.", "", pd.DataFrame()
    r = router.route(text)
    conf = f" — confidence **{r['confidence']:.2f}**" if r["confidence"] is not None else ""
    decision = (f"## ➜ {r['team']}\n"
                f"Decided by: **{r['decided_by']}**  ·  intent: `{r['intent']}`{conf}")
    if r["reply"]:
        decision += f"\n\n**Bot reply:** {r['reply']}"
    corr = ", ".join(f"`{a}` → `{b}` (edit distance {d})" for a, b, d in r["corrections"]) or "none"
    codes = ", ".join(f"`{c}`" for c in r["model_codes"]) or "none"
    pre = (f"**Normalized text:** `{r['normalized']}`  \n"
           f"**Extracted model codes:** {codes}  \n"
           f"**Spelling corrections (min. edit distance):** {corr}")
    df = pd.DataFrame(r["rows"])[["model", "top_intent", "confidence", "threshold", "decision", "team", "latency_ms"]]
    return decision, pre, df


loaded = ", ".join(DISPLAY[m] for m in DISPLAY if m in router.models) or "none"
warning = "" if PRIMARY in router.models else (
    "\n\n> ⚠️ The quantized Transformer was not found in `models/`. Run `python scripts/download_models.py` "
    "or train it with the Colab notebook.")

with gr.Blocks(title="SWE015 Ticket Router") as demo:
    gr.Markdown("# 🛠️ Customer Support Ticket Router\n"
                "Type a customer message (English or Turkish). The message is normalized, classified into one of "
                "156 intents and routed to a team by the **INT8-quantized distilled BERT**. Low-confidence or "
                "out-of-scope messages go to a human agent; small talk gets an ELIZA-style reply.  \n"
                f"Loaded models: {loaded}{warning}")
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(label="Customer message", lines=3, placeholder="e.g. my washing machine won't drain")
            btn = gr.Button("Route message", variant="primary")
            gr.Examples([[e] for e in EXAMPLES], inputs=inp)
        with gr.Column(scale=4):
            out_decision = gr.Markdown()
            out_pre = gr.Markdown()
    out_table = gr.Dataframe(label="All four models side by side (CPU, single message)", wrap=True)
    btn.click(handle, inp, [out_decision, out_pre, out_table])
    inp.submit(handle, inp, [out_decision, out_pre, out_table])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.launch(share=a.share, server_port=a.port)
