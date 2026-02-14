import gradio as gr
import torch
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load trained decision model
model = torch.load("decision_model.pt", map_location=torch.device("cpu"))
model.eval()

label_map = {
    0: "respond",
    1: "ask_clarify",
    2: "defer",
    3: "silent"
}

def evaluate(text):

    if text.strip() == "":
        return "silent", 1.0, 1.0, "noise"

    emb = embedder.encode([text], convert_to_tensor=True)

    with torch.no_grad():
        logits = model(emb)
        probs = torch.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, 1)

        sorted_probs, _ = torch.sort(probs, descending=True)
        margin = sorted_probs[0][0] - sorted_probs[0][1]

    decision = label_map[pred.item()]
    confidence = float(confidence)
    margin = float(margin)

    # Risk category mapping
    if decision == "respond":
        category = "trusted"
    elif decision == "ask_clarify":
        category = "ambiguous"
    elif decision == "defer":
        category = "high_risk"
    else:
        category = "noise"

    return decision, confidence, margin, category


demo = gr.Interface(
    fn=evaluate,
    inputs=gr.Textbox(label="User Input"),
    outputs=[
        gr.Textbox(label="Decision"),
        gr.Number(label="Confidence"),
        gr.Number(label="Decision Margin"),
        gr.Textbox(label="Risk Category")
    ],
    title="LLM Reliability Evaluation Framework",
    description="Confidence-calibrated decision gating system that evaluates whether an AI should respond, clarify, defer, or remain silent before LLM invocation."
)

demo.launch()
